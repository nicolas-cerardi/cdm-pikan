from absl import app
from absl import flags
import sys
import os
import jax
import jaxlib
print(jax.__version__)
print(jaxlib.__version__)
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
from tqdm import tqdm
from jaxkan.KAN import KAN
import jax.numpy as jnp
from jax_data import SimDataset
from jax import grad, vmap
from jaxopt.loss import huber_loss

from flax import nnx
from flax import serialization
import optax
import orbax.checkpoint as ocp

import matplotlib.pyplot as plt
import graphics as gr
import numpy as np
from jaxkan.utils.PIKAN import gradf
from jax_utils import force_1d_parallel
import pickle as pkl

FLAGS = flags.FLAGS
# Net arch
flags.DEFINE_list("layers", ["2", "8", "12", "8", "1"], "Width of each layer in the KAN.")
flags.DEFINE_integer("grid_size", 16, "Number of splines for each activation")
flags.DEFINE_float("grid_adaptivity", 1.0, "Adaptivity of the grid")
flags.DEFINE_integer("netseed", 1, "init seed for net")

# Training
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for the optimizer")
flags.DEFINE_integer("num_epochs", 100000, "Number of epochs to train")
flags.DEFINE_integer("n_pde_a", 64, "time samples in pde batch")
flags.DEFINE_list("schedule_epochs", ["80000"], "Epochs to update the learning rate")
flags.DEFINE_list("schedule_factor", ["0.1"], "Factor to reduce the learning rate")
flags.DEFINE_float("a_start", 0.91, "Initial sim time")
flags.DEFINE_float("a_end", 3.00, "Final sim time")
flags.DEFINE_integer("log", 100, "Log every n epochs")
flags.DEFINE_integer("datakey", 2, "Random seed for data generation")
flags.DEFINE_string("losstype", "PDE1", "PDE1 or PDE2")

# I/O
flags.DEFINE_string("output_path", None, "Path to save the model")
flags.DEFINE_string("jobname", "", "jobname")
flags.DEFINE_string("previous_model", None, "Path to the previous model")
flags.DEFINE_float("a_previous", None, "Initial sim time for previous model")

def make_model(layers, seed):
    '''
    Create a KAN model
    '''
    #req_params = {'k': 3, 'noise_std':0.05}
    req_params = {'k': 3, 'G':16, 'grid_e':1.0, 'grid_range':(-0.5, .5),
                'base_basis':0.05, 'base_spline':0.05, 'base_res':0.05,
                'pow_basis':0.5, 'pow_spline':0.5, 'pow_res':0.5}
    layer_type = 'spline'

    model = KAN(layer_dims = layers,
                layer_type = layer_type,
                required_parameters = req_params,
                add_bias = True,
                rngs = nnx.Rngs(seed)
            )
    return model

def process_outputname(jobstr, layers, n_pde_a):
    '''
    Process the output name
    '''
    outstr = "./run/net1d_jaxKAN_seq_%s_ndi%d_%s/"%('-'.join([str(i) for i in layers]), n_pde_a, jobstr)
    outstr = os.path.abspath(outstr)
    print(outstr)
    os.mkdir(outstr)
    return outstr

def make_ic_from_previous(checkpointer, model, previous_model, sim_data):
    #replace initial condition
    init_state = nnx.state(model)
    
    previous_model = os.path.abspath(previous_model)
    previous_state = checkpointer.restore(previous_model, item=init_state)
    # Update the model with the loaded state
    nnx.update(model, previous_state)
    pde_qa = jnp.stack([sim_data.ic_q.squeeze(), sim_data.ic_a.squeeze()]).T
    def z(x):
        return model(x)
    z0 = model(pde_qa)
    v0 = gradf(z,1,1)(pde_qa)
    acc0 = gradf(z,1,2)(pde_qa)
    #do not forget to put the initial weights back in the model
    nnx.update(model, init_state)

    return z0.squeeze(), v0.squeeze(), acc0.squeeze()
    
def myloss(x, y):
    #return jnp.mean(huber_loss(x, y))
    return jnp.mean((x - y)**2)

def eq_PDE1(this_q, this_z0, this_a, zpde, zpde_t, zpde_tt):
    x_net_d = (this_q + this_z0 + zpde).reshape((FLAGS.n_pde_a, -1, 1))
    pt_q0_d    = -0.5
    force1_net_par = -jnp.sum(x_net_d > x_net_d.transpose(0,2,1), axis=2, keepdims=True) / 8192 + this_q.reshape((FLAGS.n_pde_a, -1, 1)) - pt_q0_d
    force2_net = zpde_tt * this_a * this_a * 2 / 3 + zpde_t * this_a - zpde - this_z0
    return force1_net_par, force2_net.reshape((FLAGS.n_pde_a, -1, 1))

def eq_PDE2(this_q, this_z0, this_a, zpde, zpde_t, zpde_tt):
    x_net_d = this_q + this_z0 + zpde
    force1_net_par = force_1d_parallel(
        x_net_d.reshape((FLAGS.n_pde_a, -1, 1)), 
        this_q.reshape((FLAGS.n_pde_a, -1, 1))
    )
    force2_net = zpde_tt * this_a ** 2 * 2 / 3 + zpde_t * this_a
    return force1_net_par, force2_net.reshape((FLAGS.n_pde_a, -1, 1))


def main(argv):

    layers = [int(v) for v in FLAGS.layers]
    outstr = process_outputname(FLAGS.jobname, layers, FLAGS.n_pde_a)    
    model = make_model(layers, FLAGS.netseed)
    
    size = 'adaptdt_seq_custom%.2fto%.2f'%(FLAGS.a_start, FLAGS.a_end)
    sim_data = SimDataset(n_data_points = 8000*8000, n_pde_a=FLAGS.n_pde_a, 
                          key=FLAGS.datakey, size='largecustom0.91to3.01', supervised=True)
    print(sim_data.data.shape, sim_data.a.shape, sim_data.a[:3], sim_data.a[-3:])
    
    # Initialize the checkpointer once
    checkpointer = ocp.PyTreeCheckpointer()

    # IC initialization. Step 1 : explore the parent list
    reached_origin = (FLAGS.previous_model is None) #True for stage 1, otherwise False
    if not reached_origin:
        parent = '/'.join(FLAGS.previous_model.split('/')[:-1])
    parent_list = []
    parentdata_list = [
        {'previous_model':FLAGS.previous_model, 'a_start':FLAGS.a_start, 'a_end':FLAGS.a_end, 'datakey':FLAGS.datakey}
        ]
    while not reached_origin:
        # this means that parent is not None
        with open(parent+'/metadata.pkl', 'rb') as f:
            parent_metadata = pkl.load(f)
        parent_list.append(parent)
        parentdata_list.append(parent_metadata)
        reached_origin = (parent_metadata['previous_model'] is None)
        if parent_metadata['previous_model'] is not None: #then we can add it to the list
            parent = '/'.join(parent_metadata['previous_model'].split('/')[:-1])

    print('Parent models:', parent_list)

    # IC initialization. Step 2 : load and add up all previous displacements
    sum_previous_z0 = jnp.zeros_like(sim_data.z0.squeeze())
    for stage in range(len(parentdata_list)-1):
        # In the first stage, we need to load the true z0
        if stage == 0: #then z0 is the zeldovich approx
            size = 'adaptdt_seq_custom%.2fto%.2f'%(parentdata_list[-1-stage]['a_start'], parentdata_list[-1-stage]['a_end'])
            sim_data = SimDataset(n_data_points = 8000*8000, n_pde_a=FLAGS.n_pde_a, 
                                key=parentdata_list[-1-stage]['datakey'], size=size)
            sum_previous_z0 += sim_data.z0.squeeze()
        size = 'adaptdt_seq_custom%.2fto%.2f'%(parentdata_list[-1-stage-1]['a_start'], parentdata_list[-1-stage-1]['a_end'])
        sim_data = SimDataset(n_data_points = 8000*8000, n_pde_a=FLAGS.n_pde_a, 
                                key=parentdata_list[-1-stage]['datakey'], size=size)
        # We also need to fetch the initial conditions from the previous model
        z0_ext, v0_ext, acc0_ext = make_ic_from_previous(checkpointer, model, parentdata_list[-1-stage-1]['previous_model'], sim_data)
        print('loaded model %s at time %.2f'%(parentdata_list[-1-stage-1]['previous_model'],parentdata_list[-1-stage-1]['a_start']))
        sum_previous_z0 += z0_ext.squeeze()

    if FLAGS.previous_model is not None:
        size = 'adaptdt_seq_custom%.2fto%.2f'%(FLAGS.a_start, FLAGS.a_end)
        sim_data = SimDataset(n_data_points = 8000*8000, n_pde_a=FLAGS.n_pde_a, 
                              key=FLAGS.datakey, size=size, z0_ext=sum_previous_z0, v0_ext=v0_ext, acc0_ext=acc0_ext)
        
    print(sim_data.data.shape, sim_data.a.shape, sim_data.a[:3], sim_data.a[-3:])
    print(model)
    # Select the PDE loss function
    used_pde_func = eq_PDE1 if FLAGS.losstype == 'PDE1' else eq_PDE2

    # PDE Loss
    @nnx.jit
    def compute_loss(model, sample):
        def z(x):
            return model(x)
        
    
        ### DATA
        q_data = sample["dens_q"]
        a_data = sample["dens_a"]
        x_dens = sample["dens_z"] + q_data + sample['dens_z0'].reshape(1,-1)
        data_dens = force_1d_parallel(x_dens[..., jnp.newaxis],q_data[..., jnp.newaxis],upscale=500)

        dens_qa = jnp.stack([jnp.ravel(q_data), jnp.ravel(a_data)]).T
        zdens = model(dens_qa).reshape(2,-1)  # output of u(x,t)
        zdens_t = gradf(z,1,1)(dens_qa).reshape(2,-1) 
        zdens_tt = gradf(z,1,2)(dens_qa).reshape(2,-1) 
        x_dens_net = q_data + zdens.reshape(2,-1) + sample['dens_z0'].reshape(1,-1)
        net_dens = force_1d_parallel(x_dens_net[..., jnp.newaxis],q_data[..., jnp.newaxis],upscale=500)
        loss_ic = myloss(zdens_t, sample["dens_v"])  + myloss(data_dens, net_dens) + myloss(zdens, sample["dens_z"]) #myloss(data_dens, net_dens)
        
        ### ICs
        #ic_qa = jnp.stack([jnp.ravel(sample["data_q"]), jnp.ravel(sample["data_a"])]).T
        #zic = model(ic_qa)
        #zic_t = gradf(z,1,1)(ic_qa)
        #zic_tt = gradf(z,1,2)(ic_qa)
        #loss_ic = myloss(zic,sample["data_z"]) + myloss(zic_t, sample["data_v"])+ myloss(zic_tt, sample["data_acc"])

        ### BCs
        bc_qa = jnp.stack([jnp.ravel(sample["bc_q"]), jnp.ravel(sample["bc_a"])]).T
        zbc = model(bc_qa)  # output of u(x,t)
        zbc_t = gradf(z,1,1)(bc_qa)
        zbc_tt = gradf(z,1,2)(bc_qa)
        loss_bc = myloss(zbc, sample["bc_z"]) + myloss(zbc_t, sample["bc_v"]) + myloss(zbc_tt, sample["bc_acc"])

        ### PDE
        this_q = jnp.tile(sample["pde_q"], (1, FLAGS.n_pde_a)).T.ravel().reshape(-1, 1)
        this_z0 = jnp.tile(sample["pde_z0"], (1, FLAGS.n_pde_a)).T.ravel().reshape(-1, 1)
        this_a = sample["pde_a"].T.ravel().reshape(-1, 1)
        
        pde_qa = jnp.stack([this_q.squeeze(), this_a.squeeze()]).T
        
        zpde = model(pde_qa)
        zpde_t = gradf(z,1,1)(pde_qa)
        zpde_tt = gradf(z,1,2)(pde_qa)

        force1_net_par, force2_net = used_pde_func(this_q, this_z0, this_a, zpde, zpde_t, zpde_tt)
        #print(force1_net_par.shape, zpde_tt.shape, this_a.squeeze().shape, zpde_t.shape, force2_net.shape)
        mse_pde = myloss(force1_net_par, force2_net)/FLAGS.n_pde_a
        
        loss = loss_ic + loss_bc + mse_pde
        return loss, loss_ic, loss_bc, mse_pde
    
    # Define train loop
    @nnx.jit
    def train_step(model, optimizer, sample):

        def loss_fn(model):
            loss, _, _, _ = compute_loss(model, sample)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    # Setup scheduler for optimizer
    scheduler = optax.piecewise_constant_schedule(
        init_value=FLAGS.learning_rate,
        boundaries_and_scales={int(ep) : float(fac) for ep, fac in zip(FLAGS.schedule_epochs, FLAGS.schedule_factor)},
    )
    
    opt_type = optax.adam(learning_rate=scheduler)
    optimizer = nnx.Optimizer(model, opt_type)

    # Initialize train_losses
    train_losses = jnp.zeros((FLAGS.num_epochs//FLAGS.log,))
    ic_losses = jnp.zeros((FLAGS.num_epochs//FLAGS.log,))
    bc_losses = jnp.zeros((FLAGS.num_epochs//FLAGS.log,))
    pde_losses = jnp.zeros((FLAGS.num_epochs//FLAGS.log,))
    lr_vals = jnp.zeros((FLAGS.num_epochs//FLAGS.log,))
    train_dset = iter(sim_data)

    pbar = tqdm(range(FLAGS.num_epochs), desc='description', ncols=100)

    step_best_loss = 0
    count_bests = 0
    best_loss = 1e6

    for epoch in pbar:
        # Calculate the loss
        sample = next(train_dset)
        loss = train_step(model, optimizer, sample)

        # Append the loss
        if epoch % FLAGS.log == 0:
            loss, loss_ic, loss_bc, mse_pde = compute_loss(model, sample)
            lr = scheduler(epoch)

            pbar.set_description("loss: %.2e " % (loss))

            train_losses = train_losses.at[epoch//FLAGS.log].set(loss)
            ic_losses = ic_losses.at[epoch//FLAGS.log].set(loss_ic)
            bc_losses = bc_losses.at[epoch//FLAGS.log].set(loss_bc)
            pde_losses = pde_losses.at[epoch//FLAGS.log].set(mse_pde)
            lr_vals = lr_vals.at[epoch//FLAGS.log].set(lr)

        if epoch % 5000 == 0 and epoch>80000:
            state = nnx.state(model)
            checkpointer.save(outstr+'/epoch%d'%epoch, state)
        
        if loss < best_loss*0.99 and epoch > 10000: #require minimum 1% improvement
            best_loss = loss
            count_bests += 1
            step_best_loss = epoch
            state = nnx.state(model)
            checkpointer.save(outstr+'/best', state, force=True)

    #add a metadata file with dict structure and save with pickle
    metadata = {
        'jobname': FLAGS.jobname,
        'layers': layers,
        'grid_size': FLAGS.grid_size,
        'grid_adaptivity': FLAGS.grid_adaptivity,
        'netseed': FLAGS.netseed,
        'output_path': FLAGS.output_path,
        'previous_model': FLAGS.previous_model,
        'n_pde_a': FLAGS.n_pde_a,
        'learning_rate': FLAGS.learning_rate,
        'num_epochs': FLAGS.num_epochs,
        'a_start': FLAGS.a_start,
        'a_end': FLAGS.a_end,
        'losstype': FLAGS.losstype,
        'a_previous': FLAGS.a_previous,
        'datakey': FLAGS.datakey
    }
    with open(outstr+'/metadata.pkl', 'wb') as f:
        pkl.dump(metadata, f)


    plt.figure(figsize=(7, 4))

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.arange(0, FLAGS.num_epochs, FLAGS.log), np.array(train_losses), color='k')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Total Loss')
    ax1.set_yscale('log')
    ax1.axvline(x=step_best_loss, linestyle='--', color='g')

    ax2 = ax1.twinx()
    ax2.plot(np.arange(0, FLAGS.num_epochs, FLAGS.log), lr_vals, 'b--')
    ax2.set_ylabel('Learning Rate', color='b')
    ax2.tick_params('both', colors='b', direction='in', which='both')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(0, FLAGS.num_epochs, FLAGS.log), np.array(ic_losses), color='k')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('IC Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.subplot(2, 2, 3)
    plt.plot(np.arange(0, FLAGS.num_epochs, FLAGS.log), np.array(bc_losses), color='k')
    plt.ylabel('Loss')
    plt.title('BC Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.subplot(2, 2, 4)
    plt.plot(np.arange(0, FLAGS.num_epochs, FLAGS.log), np.array(pde_losses), color='k')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('PDE Loss')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(outstr+'/learning_curve.png')

    print('Finished training, best loss:', best_loss, 'count_bests:', count_bests, 'step_best_loss:', step_best_loss)

  


if __name__ == '__main__':
  app.run(main)
