import jax
import jax.numpy as jnp
import numpy as np
import jax.random as random

#####################
# data reading
#####################
def read_sim_data(size="large"):

    # getting training data
    # files contain 0: q 1: psi , 2: v 3: acc 4: acc_Zeldo 5: x

    a_list = [ '0.9','1.1', '1.5','2', '2.5', '3', '5' ]

    if size=="medium":
        a_list = ["{:.0f}".format(i) if i.is_integer() else "{:.1f}".format(i) for i in np.linspace(1,3,21)]
        a_list = ["{:.1f}".format(i).replace('.0','') for i in np.arange(1,2.1,0.1)]

    elif size=="large":
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(0.91,3.01,0.01)]
        a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]
    
    elif size=="largelong6":
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(0.91,6.01,0.01)]
        a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]
    
    elif size=="largelong10":
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(0.91,10.01,0.01)]
        a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]
    
    elif size=="largelong30":
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(0.91,30.01,0.01)]
        a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]
    
    elif "custom" in size:
        #example: largecustom3.00to5.00
        bounds = size.split('custom')[1].split('to')
        if 'space01' in size:
            spacing=0.1
        else:
            spacing=0.01
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(float(bounds[0]),float(bounds[1]),spacing)]
        a_list = [a.rstrip('0').rstrip('.') if '.' in a else a for a in a_list]
        #a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]
    if "adaptdt_seq" in size:
        bounds = size.split('custom')[1].split('to')
        print(bounds)
        spacing = 0.5
        if bounds[0] == '0.90':
            a_list = ['0.9'] + ["{:.2f}".format(i).replace('.00','') for i in np.arange(1.0,float(bounds[1]),spacing)]
        else:
            a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(float(bounds[0]),float(bounds[1]),spacing)]
        print(a_list)
        a_list = [a.rstrip('0').rstrip('.') if '.' in a else a for a in a_list]

    elif size=="huge":
        a_list = ["{:.2f}".format(i).replace('.00','') for i in np.arange(0.91,1.18,0.01)]
        a_list = [a[:-1] if a[-1]=='0' else a for a in a_list]

    file_end = "run_cosmo_sim_1d/output_nsteps_8192_astart_0.9_aend_{0}_c0.txt"
    if size=="huge":
        file_end = "run_cosmo_sim_1d/output_nsteps_16384_astart_0.9_aend_{0}_c0.txt"
    
    elif size=="largelong6" or 'largecustom' in size:
        file_end = "run_cosmo_sim_1d/cosmo_sim_1d_long6/output_nsteps_8192_astart_0.9_aend_{0}_c0.txt"
    
    dt = (np.log10(3.0) - np.log10(0.90))/1024 #8192 #
    print("dt", dt)
    if 'adaptcustom' in size:
        dt = (np.log10(3.0) - np.log10(0.90))/1024 #8192
        print("dt", dt)
        file_end = "run_cosmo_sim_1d/run_sim_extended_adaptnstep/output_nsteps_{0}_astart_0.9_aend_{1}_c0.txt"
    elif 'npart2048custom' in size:
        file_end = "run_cosmo_sim_1d/run_sim_extended_0.91_10.00_npart2048/output_nsteps_8192_astart_0.9_aend_{0}_c0.txt"
    elif 'npart4096custom' in size:
        file_end = "run_cosmo_sim_1d/run_sim_extended_0.91_10.00_npart4096/output_nsteps_8192_astart_0.9_aend_{0}_c0.txt"
    elif 'nstep2048custom' in size:
        file_end = "run_cosmo_sim_1d/run_sim_extended_0.91_10.00_nstep2048/output_nsteps_2048_astart_0.9_aend_{0}_c0.txt"
    elif 'nstep4096custom' in size:
        file_end = "run_cosmo_sim_1d/run_sim_extended_0.91_10.00_nstep4096/output_nsteps_4096_astart_0.9_aend_{0}_c0.txt"
    elif 'adaptdt_c00' in size:
        file_end = "run_cosmo_sim_1d/run_sim_adaptdt_c00/output_nsteps_{0}_astart_0.01_aend_{1}_c0.txt"
    elif 'adaptdt_c01' in size:
        file_end = "run_cosmo_sim_1d/run_sim_adaptdt_c01/output_nsteps_{0}_astart_0.01_aend_{1}_c0.1.txt"
    elif 'adaptdt_c04' in size:
        file_end = "run_cosmo_sim_1d/run_sim_adaptdt_c04/output_nsteps_{0}_astart_0.01_aend_{1}_c0.4.txt"
    elif 'adaptdt_seq' in size:
        file_end = "run_cosmo_sim_1d/run_sim_adaptdt_seq1024/output_nsteps_{0}_astart_0.5_aend_{1}_c0.txt"
    data_out = []

    for i,a in enumerate(a_list):
        if 'adapt' in size:
            nsteps = max([1024, int((np.log10(float(a)) - np.log10(0.90))/dt)])
            #file_end = "run_cosmo_sim_1d/run_sim_extended_adaptnstep/output_nsteps_{0}_astart_0.9_aend_{1}_c0.txt"
            data = np.loadtxt(file_end.format(nsteps, a), dtype=np.float32)
        else:
            data = np.loadtxt(file_end.format(a), dtype=np.float32)
        if i==0:
            data_out = np.zeros( (len(a_list),data.shape[1],data.shape[0]),dtype=np.float32)
        data_out[i,:,:] = data.T

    a_list = np.array([float(a) for a in a_list],dtype=np.float32)

    return data_out, a_list

def make_x(q, t, dx0):
    # read q coordinate from data file for time 0
    x_q, x_t = jnp.meshgrid(q, t) #do not + dx0
    x_dx0 = jnp.zeros_like(x_q) + dx0
    x_q, x_t, x_dx0 = x_q.flatten(), x_t.flatten(), x_dx0.flatten()
    
    return jnp.vstack([x_q, x_t, x_dx0]).astype(jnp.float32).T

def make_y(data):
    # Compute differences and extract relevant slices using JAX operations
    y_dx = data[:, 1, :] - data[0, 1, :]
    y_v = data[:, 2, :]
    y_acc = data[:, 3, :]
    
    
    # Flatten all arrays
    y_dx,y_v,y_acc = y_dx.flatten(),y_v.flatten(),y_acc.flatten()
    
    return jnp.vstack([y_dx, y_v, y_acc]).astype(jnp.float32).T

class SimDataset:
    def __init__(self,  size = "large", n_data_points = 16000, 
                 supervised=False, n_pde_q=2**13, n_pde_a=3, n_ics=500, 
                 n_data=500, a_end=None, key=0, z0_ext=None, v0_ext=None, acc0_ext=None,
                 training=True):
        self.n_pde_q = n_pde_q
        self.n_pde_a = n_pde_a
        self.n_ics = n_ics
        self.n_data = n_data
        self.supervised = supervised
        self.key = random.PRNGKey(key)

        self.data, self.a  = read_sim_data(size)
        self.shape = self.data[:,0,:].shape
        self.q =  self.data[0,0,:]

        self.data_a = self.a.copy()

        if 'adaptdt_seq' in size:
            if training:
                bounds = size.split('custom')[-1].split('to')
                self.a = jnp.linspace(float(bounds[0]), float(bounds[1]), int((float(bounds[1])-float(bounds[0]))*100)+1)
            for i, a in enumerate(self.data_a):
                self.data[i,2,:] = self.data[i,2,:]/a**1.5
                self.data[i,3,:] = (self.data[i,3,:] - self.data[i,2,:]*a)*3/2/a**2
            self.shape = (self.a.size, self.q.size)
        else:
            # need to rescale the velocity coordinate so that v = dx/da
            for i, a in enumerate(self.a):
                self.data[i,2,:] = self.data[i,2,:]/a**1.5
                self.data[i,3,:] = (self.data[i,3,:] - self.data[i,2,:]*a)*3/2/a**2
        
        if z0_ext is None:
            self.dx0 =  self.data[0,1,:]
        else:
            self.dx0 = z0_ext

        self.id_a_start = np.argmin(np.abs(self.a-0.9))
        self.a_start = self.a[self.id_a_start]
        self.a_end = self.a[-1] if a_end is None else a_end #default should be self.a[-1]

        #self.q_grid = self.data[:,0,:]
        x = make_x(self.q, self.data_a, self.dx0)
        y = make_y(self.data)
        
        x = np.vstack( [x[:,0], x[:,1], np.arange(0,x.shape[0])]).astype(np.float32).T

        indices = np.arange(x.shape[0]) #np.random.permutation(x.shape[0])
        training_idx, test_idx = indices[:n_data_points], indices[n_data_points:]
        self.x = {'train':x[training_idx,:], 'val':x[test_idx,:], 'all':x}
        self.y = {'train':y[training_idx,:], 'val':y[test_idx,:], 'all':y}
        
        #for BCs
        self.bc_q, self.bc_a, self.bc_z0, self.bc_z, self.bc_v, self.bc_acc = self.get_bcs()

        #for ICs
        if v0_ext is not None and acc0_ext is not None:
            self.ic_q, self.ic_a, self.ic_z0, self.ic_z, self.ic_v, self.ic_acc = self.indep_ics(v0_ext, acc0_ext)
        else:
            self.ic_q, self.ic_a, self.ic_z0, self.ic_z, self.ic_v, self.ic_acc = self.get_ics()

        if self.supervised:
            self.q_data = self.x['train'][:,[0]].reshape((self.a.size, -1))
            self.a_data = self.x['train'][:,[1]].reshape((self.a.size, -1))
            #self.z0_data = self.z0.reshape((self.a.size, -1))
            self.z_data = self.y['train'][:,[0]].reshape((self.a.size, -1))
            self.v_data = self.y['train'][:,[1]].reshape((self.a.size, -1))
            self.acc_data = self.y['train'][:,[2]].reshape((self.a.size, -1))
        #for PDEs
        q_density = np.linspace(-0.5,0.5,n_pde_q)
        self.q_density = q_density.reshape(-1,1)
        self.z0_density = (np.interp(self.q_density, self.q, self.z0)).reshape(-1,1)
        self.a_density = np.linspace(self.a_start, self.a_end,2**8)

        print(" -- Dataset prepared -- ")
        print("a is from %.2f to %.2f" % (self.a_start, self.a_end))
        print("ic, bc, data and pde shapes are:", self.ic_q.shape, self.bc_q.shape, self.q_density.shape)
        if self.supervised:
            print("data shape is:", self.q_data.shape)
        #self.x_train_dataset = np.vstack( [self.x_train[:,0],self.x_train[:,1],self.training_idx],dtype=np.float32 ).T
        #self.x_val_dataset   = np.vstack( [self.x_val[:,0],self.x_val[:,1],self.test_idx],dtype=np.float32 ).T
    
    @property
    def z0(self):
        return self.dx0
    
    @property
    def z(self):
        return self.data[:,1,:] - self.z0# - self.data[0,1,:] nope
    @property
    def v(self):
        return self.data[:,2,:]
    @property
    def acc(self):
        #acc = np.zeros(self.z.shape)
        #for i,a in enumerate(self.a):
        #    acc[i,:] = (self.data[i,3,:] - self.data[i,2,:]*a)*3/2/a**2
        return self.data[:,3,:]
    @property
    def F(self):
        print("DO NOT USE ME")
        return self.data[:,3,:]
    
    def get_ics(self): #do not +self.dx0.reshape(-1,1)
        ic_q = self.q.reshape(-1,1)
        ic_a = np.full((len(self.q),1), self.a[0])
        ic_z0 = self.z0.reshape(-1,1)
        ic_z = self.z[0,:].reshape(-1,1) #should be 0
        ic_v = self.v[0,:].reshape(-1,1)
        ic_acc = self.acc[0,:].reshape(-1,1)
        return ic_q, ic_a, ic_z0, ic_z, ic_v, ic_acc
    
    def indep_ics(self, v0_ext, acc0_ext): #do not +self.dx0.reshape(-1,1)
        ic_q = self.q.reshape(-1,1)
        ic_a = np.full((len(self.q),1), self.a[0])
        ic_z0 = self.z0.reshape(-1,1)
        ic_z = jnp.zeros_like(ic_z0)
        ic_v = v0_ext.reshape(-1,1)
        ic_acc = acc0_ext.reshape(-1,1)
        return ic_q, ic_a, ic_z0, ic_z, ic_v, ic_acc
    
    def get_bcs(self):
        bc_a,bc_q = np.meshgrid(self.a,[-0.5, 0.5])
        bc_a = bc_a.reshape(-1,1)
        bc_q = bc_q.reshape(-1,1)
        bc_z0 = jnp.zeros_like(bc_q)
        bc_z = jnp.zeros_like(bc_q)
        bc_v = jnp.zeros_like(bc_q)
        bc_acc = jnp.zeros_like(bc_q)
        return bc_q, bc_a, bc_z0, bc_z, bc_v, bc_acc

    def __getitem__(self, idx):
        '''Return all bcs, (n_ics,) ICs, (n_q_pde, n_a_pde) points, and n_data data'''
        key, key_ic, key_pde, key_data = random.split(self.key, 4)
        self.key = key

        # sample n_ics random ICs
        idx_ti = random.choice(key_ic, self.ic_q.shape[0], (self.n_ics,), replace=False)
        
        # BCs already ready

        # PDE
        pde_a_d = random.choice(key_pde, self.a_density, (self.n_pde_a,), replace=False).reshape((1, self.n_pde_a))
        pt_q_d = self.q_density
        pt_z0_d = self.z0_density
        pt_a_d = jnp.tile(pde_a_d, (self.n_pde_q, 1))
        
        sample = { #do not + self.z_ic[idx_ti]
            "ic_q": self.ic_q[idx_ti] , "ic_a": self.ic_a[idx_ti], "ic_z0": self.ic_z0[idx_ti],
            "ic_z": self.ic_z[idx_ti], "ic_v": self.ic_v[idx_ti], #do not jnp.zeros_like(self.z_ic[idx_ti])
            "ic_acc": self.ic_acc[idx_ti],
            "bc_q": self.bc_q, "bc_a": self.bc_a, "bc_z0": self.bc_z0,
            "bc_z": self.bc_z, "bc_v": self.bc_v, "bc_acc": self.bc_acc,

            "pde_q": pt_q_d , "pde_a": pt_a_d, "pde_z0": pt_z0_d #do not + pt_z0_d
        }
        
        if self.supervised:
            id_data = random.choice(key_data, self.q_data.shape[1], (self.n_data,), replace=False)
            pt_q = self.q_data[-1,idx_ti]
            pt_a = self.a_data[-1,idx_ti]
            pt_z = self.z_data[-1,idx_ti]
            pt_z0 = self.z0[idx_ti]
            pt_v = self.v_data[-1,idx_ti]
            pt_acc = self.acc_data[-1,idx_ti]
            
            sample['data_q'] = pt_q.reshape(-1,1)
            sample['data_a'] = pt_a.reshape(-1,1)
            sample['data_z'] = pt_z.reshape(-1,1)
            sample['data_z0'] = pt_z0.reshape(-1,1)
            sample['data_v'] = pt_v.reshape(-1,1)
            sample['data_acc'] = pt_acc.reshape(-1,1)
            #-#-#-#-#
            sample['dens_z'] = self.z_data[-2:,:]
            sample['dens_v'] = self.v_data[-2:,:]
            sample['dens_q'] = self.q_data[-2:,:]
            sample['dens_acc'] = self.acc_data[-2:,:]
            sample['dens_a'] = self.a_data[-2:,:]
            sample['dens_z0'] = self.z0
        
        return sample