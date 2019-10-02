#Python standard


#Third party
import numpy as np
import rebound
import spiceypy as spice

#Local
from . import functions
from .functions import AU

spice.furnsh("./spice/MetaK.txt")

from tqdm import tqdm

class ReboundMinDist(object):

    def __init__(self,
                integrator='IAS15',
                time_step=3600.0,
                mjd0 = 53005,
                min_MinDist = 0.01*AU,
            ):
        self.min_MinDist = min_MinDist
        self.planets = ['Mercury','Venus','Earth','Mars','Jupiter','Saturn','Uranus','Neptune']
        self.planets_mass = [0.330104e24, 4.86732e24, 5.97219e24, 0.641693e24, 1898.13e24, 568.319e24, 86.8103e24, 102.410e24]
        self.m_sun = 1.98855e30
        self._earth_ind = 3
        self.N_massive = len(self.planets) + 1;
        self.integrator = integrator
        self.time_step = time_step
        self.mjd0 = mjd0
        self.states = np.empty((6,0), dtype=np.float64)
        self.t0 = 0.0


    def _setup_sim(self):
        self.sim = rebound.Simulation()
        self.sim.units = ('m', 's', 'kg')
        self.sim.integrator = self.integrator
        self.et = functions.mjd_to_j2000(self.mjd0)*3600.0*24.0
        
        self.sim.add(m=self.m_sun)
        for i in range(0,len(self.planets)):
            #Units are always km and km/sec.
            state, lightTime = spice.spkezr(
                self.planets[i] + ' BARYCENTER',
                self.et,
                'J2000',
                'NONE',
                'SUN',
            )
            self.sim.add(m=self.planets_mass[i],
                x=state[0]*1e3,  y=state[1]*1e3,  z=state[2]*1e3,
                vx=state[3]*1e3, vy=state[4]*1e3, vz=state[5]*1e3,
            )
        self.sim.N_active = self.N_massive
        self.sim.dt = self.time_step
        
        for ind in range(self.num):
            
            x, y, z, vx, vy, vz = self.states[:,ind]
    
            self.sim.add(
                x = x,
                y = y,
                z = z,
                vx = vx,
                vy = vy,
                vz = vz,
                m = 0.0,
            )

    @property
    def num(self):
        return self.states.shape[1]
    

    def _get_state(self, ind):
        particle = self.sim.particles[self.N_massive + ind]
        state = np.empty((6,), dtype=np.float64)
        state[0] = particle.x
        state[1] = particle.y
        state[2] = particle.z
        state[3] = particle.vx
        state[4] = particle.vy
        state[5] = particle.vz
        return state

    def _get_earth_state(self):
        earth_state = np.empty((6,), dtype=np.float64)
        earth = self.sim.particles[self._earth_ind]
        earth_state[0] = earth.x
        earth_state[1] = earth.y
        earth_state[2] = earth.z
        earth_state[3] = earth.vx
        earth_state[4] = earth.vy
        earth_state[5] = earth.vz
        return earth_state

    def propagate(self, t, **kwargs):

        self._setup_sim()

        results = []
        for ind in range(self.num):
            results.append({
                'MinDist': None,
                't': None,
                'ind': None,
                'state': None,
                'state_E': None,
            })

        self.sim.move_to_com()

        states = np.empty((len(t), 6, self.num))
        states_E = np.empty((len(t), 6))

        for ti in tqdm(range(len(t))):

            self.sim.integrate(t[ti])
            
            state_E = self._get_earth_state()
            states_E[ti, :] = state_E

            for ind in range(self.num):
                
                state = self._get_state(ind)
                states[ti, :, ind] = state

                
                MinDist = np.linalg.norm(state[:3] - state_E[:3])
            
                tm = results[ind]['t']
                MinDist0 = results[ind]['MinDist']

                if tm is None or MinDist0 is None:
                    results[ind]['t'] = t[ti]
                    results[ind]['ind'] = ti
                    results[ind]['MinDist'] = MinDist
                    results[ind]['state'] = state
                    results[ind]['state_E'] = state_E
                elif MinDist < MinDist0 and MinDist0 > self.min_MinDist:
                    results[ind]['t'] = t[ti]
                    results[ind]['ind'] = ti
                    results[ind]['MinDist'] = MinDist
                    results[ind]['state'] = state
                    results[ind]['state_E'] = state_E


        return results
