import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap

def convert_pos_to_index(x,length): 
    int_pos = jnp.astype(((x+0.5)*length), jnp.int64)
    #print(int_pos.shape, int_pos.dtype)
    int_pos = jnp.remainder(int_pos, length) #or use torch.remainder ?
    return int_pos

def scatt_reduce(b,x):
    return b.at[x].add(1)

def gather(b, x):
    return b.at[x].get()

def force_1d_parallel(x, q, upscale=500):
    length = q.shape[1] * upscale
    bins = jnp.zeros((q.shape[0], length), dtype=jnp.int64)
    
    xi = convert_pos_to_index(x[:,:,0], length)
    #print(bins.shape, xi.shape)
    #density = bins.scatter_reduce(1, xi, torch.ones((n_di, q.shape[1])).long(), reduce='sum').float()
    density = vmap(scatt_reduce)(bins,xi)
    #print(density.shape, density.dtype, bins.dtype, xi.shape, xi.dtype)
    #density = density[0:length].float()
    density -= jnp.mean(density, axis=-1, keepdims=True)
    int_d = -jnp.cumsum(density, axis=1)
    #print(int_d.shape, int_d.dtype)
    return vmap(gather)(int_d, xi)[...,jnp.newaxis]/(length/upscale)
    #jnp.gather(int_d, 1, xi).unsqueeze(2)/(length/upscale) #[:,xi[0,:].long()].unsqueeze(2)/(length/upscale)

def density_1d_parallel(x,q,upscale=500):
    length = q.shape[1] * upscale
    bins = jnp.zeros((q.shape[0], length), dtype=jnp.int64)
    xi = convert_pos_to_index(x[:,:,0], length)
    density = vmap(scatt_reduce)(bins,xi)
    density -= jnp.mean(density, axis=-1, keepdims=True)
    return vmap(gather)(density, xi)[...,jnp.newaxis]/(length/upscale)
    