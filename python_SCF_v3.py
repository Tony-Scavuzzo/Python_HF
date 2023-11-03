import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import filters as sci_fil
from scipy import integrate as sci_int
import sys
import time
import copy

periodic_table = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}

# calculated using slaters rules. Note that z = Z*/n
exponent_table = {'H': 1.0, 'He': 1.7, 'Li': (2.7, 0.650), 'Be': (3.7, 0.975), 'B': (4.7, 1.300), 'C': (5.7, 1.625),
            'N': (6.7, 1.950), 'O': (7.7, 2.275), 'F': (8.7, 2.600), 'Ne': (9.7, 2.925)}


class LoadingBar():
    def __init__(self, complete):
        self.complete = complete
        self.progress = 0
        self.n_asterisks = 0
        self.time_0 = time.time()
        self.time_i = time.time()
        self.display()

    def update(self, amount):
        self.progress += amount
        new_asterisks = int(np.floor(self.progress / self.complete * 50))
        if new_asterisks > self.n_asterisks:
            self.n_asterisks = new_asterisks
            self.time_i = time.time()
            self.display()

    def display(self):
        time = self.time_i - self.time_0
        if self.n_asterisks == 50:
            print(f'[{"*" * self.n_asterisks}{" " * (50 - self.n_asterisks)}] {time:.1f} s')
        else:
            print(f'[{"*" * self.n_asterisks}{" " * (50 - self.n_asterisks)}] {time:.1f} s', end='\r')


def read_file(filename):
    # shamelessly stealing my favorite things from gaussian and orca
    # so far ignores lines with no ! and not [xyz:, :xyz]

    try:
        with open(filename) as file:
            lines = file.readlines()
    except FileNotFoundError:
        print('Error! File not found.')
        exit()

    commands = []
    coulomb_cs = []
    mo_list = []
    xyz_start = None
    xyz_end = None

    for i in range(0, len(lines)):
        lines[i] = lines[i].strip().lower()

        if lines[i].startswith('!'):
            commands.extend(lines[i].split()[1:])
        elif lines[i].startswith('ccs:'):
            coulomb_cs.append(lines[i].split()[1:])
        elif lines[i].startswith('mos:'):
            mo_list.extend(lines[i].split()[1:])
        elif lines[i] == 'xyz:':
            if xyz_start is None:
                xyz_start = i
            else:
                print('Error! xyz_start should only occur once.')
                exit()
        elif lines[i] == ':xyz':
            if xyz_end is None:
                xyz_end = i
            else:
                print('Error! xyz_end should only occur once.')
                exit()

    if xyz_start is None:
        print('Error! "xyz:" should occur exactly once.')
        exit()

    charge, mult = int(lines[xyz_start + 1].split()[0]), int(lines[xyz_start + 1].split()[1])
    if xyz_end is None:
        xyz_data = lines[xyz_start + 2:]
    else:
        xyz_data = lines[xyz_start + 2:xyz_end]

    processed_xyz_data = [item.split() for item in xyz_data if item != '']
    for row in processed_xyz_data:
        row[0] = row[0].capitalize()

    processed_commands = [command.split('=') for command in commands]

    split_mo_list = [mo.split(',') for mo in mo_list]
    processed_mo_list = [[int(coefficient) for coefficient in mo] for mo in split_mo_list]

    processed_coulomb_cs = []
    for cs in coulomb_cs:
        coordinates = []
        for i in range(0, len(cs)):
            if cs[i] == '-mos':
                mos = [int(cs[i + 1]), int(cs[i + 2])]
            elif cs[i] == '-xyz':
                coordinates.append([float(cs[i + 1]), float(cs[i + 2]), float(cs[i + 3])])
        processed_coulomb_cs.append([mos, coordinates])

    return processed_commands, processed_xyz_data, charge, mult, processed_mo_list, processed_coulomb_cs


def create_atoms(xyz_data):
    atoms_time_start = time.time()
    atoms = []
    for i in range(0, len(xyz_data)):
        symbol, x0, y0, z0 = xyz_data[i][0], float(xyz_data[i][1]), float(xyz_data[i][2]), float(xyz_data[i][3])
        x1, y1, z1 = find_nearest_gridpoint(x0, y0, z0, nuclear_gridspace)
        log.append(f'snapping {symbol}{i} to ({x1},{y1},{z1})')
        atoms.append(Nucleus(symbol, i, x1, y1, z1))
    atoms_time_end = time.time()
    if VERBOSE:
        log.append(f'atoms_time = {(atoms_time_end - atoms_time_start):.1f}')
    return atoms


def create_aos(nucleii):

    all_a_ao_time_start = time.time()
    atomic_orbitals = []
    for nucleus in nucleii:
        atomic_orbitals.extend(nucleus.create_aos('a'))
        if VERBOSE:
            log.append(f'set up {nucleus.name} orbitals')
    all_a_ao_time_end = time.time()
    if VERBOSE:
        log.append(f'set up all a orbitals')
        log.append(f'all_a_ao_time = {all_a_ao_time_end - all_a_ao_time_start}')

    if PLOTS:
        for orbital in atomic_orbitals:
            save_plot(orbital.phi, orbital.name)

    # Legacy code: I decided to focus on RHF and ROHF, so I shouldn't need beta orbitals at all
    # all_b_ao_time_start = time.time()
    # atomic_orbitals_b = copy.deepcopy(atomic_orbitals)
    # for orbital in atomic_orbitals_b:
    #     orbital.switch_to_b()
    # all_b_ao_time_end = time.time()
    # if VERBOSE:
    #     log.append(f'all_b_ao_time = {all_b_ao_time_end - all_b_ao_time_start}')

    return atomic_orbitals


def find_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


def integrate(np_array):
    # someday may need to be replaced with scipy.integrate.quad or scipy.integrate.nquad
    # return np.sum(np_array[1:-1, 1:-1, 1:-1]) * GRID ** 3
    return np.trapz(np.trapz(np.trapz(np_array[1:-1, 1:-1, 1:-1], dx=GRID, axis=0), dx=GRID, axis=0), dx=GRID, axis=0)


def calculate_nuclear_repulsion(nucleii):
    nuclear_repulsion_start = time.time()
    total_NN_energy = 0
    for i in range(0, len(nucleii)):
        for j in range(i + 1, len(nucleii)):
            A1, A2 = nucleii[i], nucleii[j]
            R = np.sqrt((A1.x - A2.x) ** 2 + (A1.y - A2.y) ** 2 + (A1.z - A2.z) ** 2)
            total_NN_energy += A1.charge * A2.charge / R
    nuclear_repulsion_end = time.time()
    if VERBOSE:
        log.append(f'nuclear_repulsion = {total_NN_energy}')
        log.append(f'nuclear_repulsion_time = {nuclear_repulsion_end - nuclear_repulsion_start}')
    return total_NN_energy


def calculate_total_onee_energy(occupied_orbitals, nucleii):
    total_onee_time_start = time.time()
    total_onee = 0
    for orbital in occupied_orbitals:
        orbital.integrate_onee(nucleii)
        total_onee += orbital.e
    total_onee_time_end = time.time()
    if VERBOSE:
        log.append(f'total_onee = {total_onee}')
        log.append(f'total_onee_time = {total_onee_time_end - total_onee_time_start}')
    return total_onee


def integrate_coulomb_term(orbital_1, orbital_2, method='1', plots=None):
    coulomb_time_start = time.time()
    if VERBOSE:
        print('starting coulomb integration')
        print('this might take a while')

    # indices of where requested plots exists
    if plots is not None:
        plot_indices = []
        for plot in plots:
            plot_indices.append(list(find_nearest_gridpoint(plot[0], plot[1], plot[2],
                                                            gridspace=electron_gridspace, type='index')))
    else:
        plot_indices = None
    log.append(f'requested plot indices: {plot_indices}')

    def phi1(x1, y1, z1):
        return orbital_1.phi[x1][y1][z1]

    def phi2(x2, y2, z2):
        return orbital_2.phi[x2, y2, z2]

    if method == '1':
        # the hard way:
        # this has been tested to be equivalent to and slower than method 2
        if VERBOSE:
            coulomb_loading_bar = LoadingBar((len(x)-2)**2)
        integral = 0
        for x1 in range(1, len(x) - 1):
            for y1 in range(1, len(y) - 1):
                for z1 in range(1, len(z) - 1):
                    for x2 in range(1, len(x) - 1):
                        for y2 in range(1, len(y) - 1):
                            for z2 in range(1, len(z) - 1):
                                # adding GRID/2 changes it to a right handed rieman sums
                                r = find_distance(x1, x2, y1, y2, z1, z2) + GRID/2
                                integral += phi1(x1, y1, z1)**2 * phi2(x2, y2, z2)**2 / r * GRID**6
                if VERBOSE:
                    coulomb_loading_bar.update(1)

    if method == '2':
        # this has been tested to be equivalent to and faster than method 1
        if VERBOSE:
            coulomb_loading_bar = LoadingBar((len(x)-2) ** 2)

        three_d_cross_section = np.zeros_like(x)
        requested_cs = []

        # traverses indices, not x
        for i in range(1, len(x) - 1):
            for j in range(1, len(y) - 1):
                for k in range(1, len(z) - 1):
                    # now that x1, y1, and z1 are established, in principle, a 3d surface can be created and integrated
                    # convert from index to value
                    a = x_range[i]
                    b = y_range[j]
                    c = z_range[k]
                    # added GRID / 2 to account for /0 error when taking 1/r
                    # As GRID -> 0, this error also -> 0
                    # if one thinks of the volume defined by 8 closest grid points, adding GRID/2 to the radius does not
                    # simply extends the point being evaluated from the corner of this cube closer to the center
                    r = find_distance(x, y, z, a, b, c) + GRID / 2

                    cross_section = phi1(i, j, k) ** 2 * orbital_2.phi ** 2 / r
                    this_integral = integrate(cross_section)

                    if plot_indices is not None:
                        if [i, j, k] in plot_indices:
                            name = f'{orbital_1.number}_{a:.2f}_{b:.2f}_{c:.2f}_{orbital_2.number}_coulomb_' \
                                   f'{this_integral:.4f}.png'
                            requested_cs.append([name, copy.deepcopy(cross_section)])
                            log.append(f'found a ccs! plotting {i}{j}{k}')

                    three_d_cross_section[i, j, k] = this_integral

                if VERBOSE:
                    coulomb_loading_bar.update(1)
        integral = integrate(three_d_cross_section)

        for cs in requested_cs:
            save_plot(cs[1], cs[0], 'med')
        save_plot(three_d_cross_section, 'coulomb_cross_section_final', 'med')

    coulomb_time_end = time.time()
    if VERBOSE:
        log.append(f'coulomb_term = {integral}')
        log.append(f'coulomb_time = {coulomb_time_end - coulomb_time_start}')

    return integral


def integrate_exchange_term(orbital_1, orbital_2, plots=None):
    exchange_time_start = time.time()
    if VERBOSE:
        print('starting exchange integration')
        print('this might take a while')

    # indices of where requested plots exists
    if plots is not None:
        plot_indices = []
        for plot in plots:
            plot_indices.append(list(find_nearest_gridpoint(plot[0], plot[1], plot[2],
                                                            gridspace=electron_gridspace, type='index')))
    else:
        plot_indices = None
    log.append(f'requested plot indices: {plot_indices}')

    def phi1(x1, y1, z1):
        return orbital_1.phi[x1][y1][z1]

    def phi2(x2, y2, z2):
        return orbital_2.phi[x2, y2, z2]

    if VERBOSE:
        exchange_loading_bar = LoadingBar((len(x)-2) ** 2)

    three_d_cross_section = np.zeros_like(x)
    requested_cs = []

    # traverses indices, not xyz
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            for k in range(1, len(z) - 1):
                # now that x1, y1, and z1 are established, in principle, a 3d surface can be created and integrated
                # convert from index to value
                a = x_range[i]
                b = y_range[j]
                c = z_range[k]
                # added GRID / 2 to account for /0 error when taking 1/r
                # As GRID -> 0, this error also -> 0
                # if one thinks of the volume defined by 8 closest grid points, adding GRID/2 to the radius does not
                # simply extends the point being evaluated from the corner of this cube closer to the center
                r = find_distance(x, y, z, a, b, c) + GRID / 2

                cross_section = phi1(i, j, k) * phi2(i, j, k) * orbital_1.phi * orbital_2.phi / r
                this_integral = integrate(cross_section)

                if plot_indices is not None:
                    if [i, j, k] in plot_indices:
                        name = f'{orbital_1.number}_{a:.2f}_{b:.2f}_{c:.2f}_{orbital_2.number}_coulomb_' \
                               f'{this_integral:.4f}.png'
                        requested_cs.append([name, copy.deepcopy(cross_section)])
                        log.append(f'found a ccs! plotting {i}{j}{k}')

                three_d_cross_section[i, j, k] = this_integral

            if VERBOSE:
                exchange_loading_bar.update(1)

    integral = integrate(three_d_cross_section)

    for cs in requested_cs:
        save_plot(cs[1], cs[0], 'med')
    save_plot(three_d_cross_section, 'coulomb_cross_section_final', 'med')

    exchange_time_end = time.time()
    if VERBOSE:
        log.append(f'exchange_term = {integral}')
        log.append(f'exchange_time = {exchange_time_end - exchange_time_start}')

    return integral


def calculate_total_coulomb_energy(orbitals, pass_method='1', plots=None):

    total_coulomb_time_start = time.time()
    total_ee_repulsion = 0
    for i in range(0, len(orbitals)):
        for j in range(i + 1, len(orbitals)):

            # checks plots to see if this set of orbitals are being plotted
            # otherwise passes None
            coordinates = None
            if plots is not None:
                for k in range(0, len(plots)):
                    if [i, j] == plots[k][0]:
                        coordinates = plots[k][1]


            total_ee_repulsion += integrate_coulomb_term(orbitals[i], orbitals[j],
                                                         method=pass_method, plots=coordinates)

    total_coulomb_time_end = time.time()
    if VERBOSE:
        log.append(f'electron_electron_repulsion = {total_ee_repulsion}')
        log.append(f'total_coulomb_time = {total_coulomb_time_end - total_coulomb_time_start}')
    return total_ee_repulsion


def save_plot(function, name, scale='abs'):
    """Creates a plot of the data as a cross section at z = 0"""

    if scale == 'abs':
        if abs(function.max()) > abs(function.min()):
            scale = abs(function.max())
        else:
            scale = abs(function.min())
    elif scale == 'med':
        scale = np.median(function) * 100
    elif scale.isdigit():
        scale = float(scale)
    elif type(scale) != float and type(scale) != int:
        print(f'Error! unrecognized scale type {scale}')

    # Define a colormap that distinguishes negative and positive values
    custom_cmap = plt.get_cmap('coolwarm')  # You can choose another colormap if preferred
    norm = colors.Normalize(vmin=-scale, vmax=scale)

    plt.figure(figsize=(6, 6))
    # plt.imshow(function[:, :, np.argmax(z_range) // 2]
    plt.imshow(function[:, :, int(SIZE//GRID)], extent=[-SIZE, SIZE, -SIZE, SIZE], origin='lower',
               cmap=custom_cmap, norm=norm)  # Use custom colormap and normalization
    plt.colorbar(label='Amplitude')
    plt.title('XY Cross-Section')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Save the plot as an image file (e.g., PNG)
    plt.savefig(name, dpi=300, bbox_inches='tight')

    # Close the plot to prevent it from being displayed in a window
    plt.close()


def sample_radial_function(function, sample_res):
    sample_step = int(sample_res//GRID)
    if sample_step < 1:
        sample_step = 1
    indices = range(0, len(x_range), sample_step)
    for index in indices:
        print(round(x_range[index], 2), function[index, len(x_range)//2, len(x_range)//2])


class Nucleus:
    # TODO: make a radius array that is passed to Slater rather than make Slater recalculate it every time

    def __init__(self, symbol, number, x0, y0, z0):
        self.symbol = symbol
        self.number = number
        self.name = f'{symbol}{number}'
        self.x = float(x0)
        self.y = float(y0)
        self.z = float(z0)

        if self.symbol in periodic_table:
            self.charge = periodic_table[self.symbol]
        else:
            print(f'Error! {self.symbol} not in periodic table!')
            exit()

        if VERBOSE:
            log.append(f'created atom {self.name}')

    def create_aos(self, spin):
        exponents = exponent_table[self.symbol]

        if type(exponents) == float:
            return [Slater(self.name, 1, 's', spin, exponents, self.x, self.y, self.z)]
        elif len(exponents) == 2:
            return [Slater(self.name, 1, 's', spin, exponents[0], self.x, self.y, self.z),
                    Slater(self.name, 2, 's', spin, exponents[1], self.x, self.y, self.z),
                    Slater(self.name, 2, 'px', spin, exponents[1], self.x, self.y, self.z),
                    Slater(self.name, 2, 'py', spin, exponents[1], self.x, self.y, self.z),
                    Slater(self.name, 2, 'pz', spin, exponents[1], self.x, self.y, self.z)]
        else:
            print("I'm not really sure how we got here")
            exit()


class Slater:
    """A single slater atomic orbital
    Right now there is only support up to p orbitals"""

    def __init__(self, atom_name, n, type, spin, exponent, x0, y0, z0):
        self.type = type

        X = x - x0
        Y = y - y0
        Z = z - z0
        R = np.sqrt(X**2 + Y**2 + Z**2)

        self.n = n
        self.type = type
        self.spin = spin
        self.name = f'{atom_name}_{n}{type}_{spin}'

        if self.n == 1 and self.type == 's':
            self.phi = np.exp(-exponent * R)
        elif self.n == 2 and self.type == 's':
            self.phi = R * np.exp(-exponent * R)
        elif self.n == 2 and self.type == 'px':
            self.phi = X * np.exp(-exponent * R)
        elif self.n == 2 and self.type == 'py':
            self.phi = Y * np.exp(-exponent * R)
        elif self.n == 2 and self.type == 'pz':
            self.phi = Z * np.exp(-exponent * R)
        else:
            raise ValueError('Error: Unrecognized orbital type')

        self.density = None
        self.normalized = False

    def switch_to_b(self):
        self.spin = 'b'
        self.name = self.name[:-1] + 'b'

    def calculate_density(self):
        self.density = self.phi**2

    def normalize(self):
        # no error handling for not normalizing first!!
        normalize_time_start = time.time()
        if self.density is None:
            self.calculate_density()

        integral = integrate(self.density)
        norm = 1 / np.sqrt(integral)

        self.phi = self.phi * norm
        self.density = self.density * norm**2
        self.normalized = True
        normalize_time_end = time.time()

        if VERBOSE:
            log.append(f'normalized! norm = {norm}')
            log.append(f'normalize_time = {normalize_time_end - normalize_time_start}')

    def plot_ao(self):
        save_plot(self.phi, f'MO{self.name}.png')
        save_plot(self.density, f'MO{self.name}_density.png')


class MO:
    def __init__(self, number, spin, atomic_orbitals, coefficients):
        self.number = number
        self.spin = spin
        self.coefficients = coefficients

        self.phi = np.zeros_like(x)
        for ao, coefficient in zip(atomic_orbitals, self.coefficients):
            self.phi += coefficient * ao.phi

        self.density = None
        self.normalized = False
        self.laplacian = None
        self.nuclear_attraction = None
        self.onee_function = None
        self.e = None

        # these attributes are only needed for verbose calculations
        self.KE_function = None
        self.KE = None
        self.PE_function = None
        self.PE = None

        # anal attributes are mostly for troubleshooting and requires special cases (H1S at origin)
        self.anal_laplacian = None
        self.anal_KE_function = None
        self.anal_KE = None

        # this attribute is not used so far, but is retained since I might need it someday
        self.gradient = None

    def calculate_density(self):
        self.density = self.phi**2

    def normalize(self):
        # no error handling for not normalizing first!!
        normalize_time_start = time.time()
        if self.density is None:
            self.calculate_density()

        integral = integrate(self.density)
        norm = 1 / np.sqrt(integral)

        self.phi = self.phi * norm
        self.density = self.density * norm**2
        self.normalized = True
        normalize_time_end = time.time()

        if VERBOSE:
            log.append(f'normalized! norm = {norm}')
            log.append(f'normalize_time = {normalize_time_end - normalize_time_start}')
        if PLOTS:
            save_plot(self.phi, f'MO{self.number}.png')
            save_plot(self.density, f'MO{self.number}_density.png')

    def calculate_gradient(self):
        # these may be off by a factor of GRID, GIRD**2, or GRID**3
        d_dx = np.gradient(self.phi, axis=0, edge_order=2)
        d_dy = np.gradient(self.phi, axis=1, edge_order=2)
        d_dz = np.gradient(self.phi, axis=2, edge_order=2)
        self.gradient = d_dx, d_dy, d_dz

    def calculate_laplacian(self):
        # laplacian calculation from filters makes bad assumptions about gird size; / GRID**2 corrects this
        laplacian_time_start = time.time()
        self.laplacian = sci_fil.laplace(self.phi) / GRID**2
        laplacian_time_end = time.time()
        if VERBOSE:
            log.append(f'laplacian_time = {laplacian_time_end - laplacian_time_start}')

    def calculate_verbose_ke(self):
        # this function assumes verbose
        if self.laplacian is None:
            self.calculate_laplacian()

        self.KE_function = -0.5 * self.phi * self.laplacian
        if PLOTS:
            save_plot(self.KE_function, f'MO{self.number}_KE.png', 'med')
        self.KE = integrate(self.KE_function)
        log.append(f'KE = {self.KE}')

    def calculate_nuclear_attraction(self, nucleii):
        nuclear_attraction_time_start = time.time()
        self.nuclear_attraction = np.zeros_like(x)
        for nucleus in nucleii:
            # avoiding the cusp by virtue of the nucleus placement
            R = np.sqrt((x - nucleus.x)**2 + (y - nucleus.y)**2 + (z - nucleus.z)**2)
            self.nuclear_attraction -= (nucleus.charge / R) * self.phi
        nuclear_attraction_time_end = time.time()
        if VERBOSE:
            log.append(f'nuclear_attraction_time = {nuclear_attraction_time_end - nuclear_attraction_time_start}')
            # sample_radial_function(self.nuclear_attraction, GRID)

    def calculate_verbose_pe(self, nucleii):
        # this function assumes verbose
        if self.nuclear_attraction is None:
            self.calculate_nuclear_attraction(nucleii)

        self.PE_function = self.phi * self.nuclear_attraction
        if PLOTS:
            save_plot(self.PE_function, f'MO{self.number}_PE.png', 'med')
        self.PE = integrate(self.PE_function)
        log.append(f'PE = {self.PE}')

    def integrate_onee(self, nucleii):
        # this function largely assumes that no other MO functions are used, using only basic error handling.
        if not VERBOSE:
            # high efficiency method
            if not self.normalized:
                self.normalize()
            if self.laplacian is None:
                self.calculate_laplacian()
            if self.nuclear_attraction is None:
                self.calculate_nuclear_attraction(nucleii)

            self.onee_function = self.phi * ((-0.5) * self.laplacian + self.nuclear_attraction)
            self.e = integrate(self.onee_function)

        else:
            if not self.normalized:
                self.normalize()
            self.calculate_verbose_ke()
            self.calculate_verbose_pe(nucleii)

            self.onee_function = self.KE_function + self.PE_function
            if PLOTS:
                save_plot(self.onee_function, f'MO{self.number}_onee.png', 'med')
            self.e = self.KE + self.PE
            log.append(f'e = {self.e}')

    # anal functions are mostly for troubleshooting, and require special cases
    def calculate_anal_laplacian(self):
        r = np.sqrt(x**2 + y**2 + z**2)
        self.anal_laplacian = ((r - 2)/r) * self.phi
        self.anal_KE_function = -0.5 * self.phi * self.anal_laplacian

    def integrate_anal_ke(self):
        if self.anal_KE_function is None:
            self.calculate_anal_laplacian()
        self.anal_KE = integrate(self.anal_KE_function)


def calculate_a_h_b(ao_a, ao_b, nuclei):
    """ao_a and ao_b are orbital objects,
     and nucleii is a list of nucleus objects.
     This function returns an element of the h matrix as a float

    This function calculates the <ao_a|h|ao_b> term of the fock matrix
    This function allows for cross terms - for the h term of
    an orbital with itself (which is used for the energy),
    pass the same orbital twice - this is the diagonal of the h matrix."""

    # TODO: incorporate gaussian methods

    def calculate_nuclear_attraction(ao_b, nuclei):
        nuclear_attraction = np.zeros_like(x)
        for nucleus in nuclei:
            # avoiding the cusp by virtue of the nucleus placement
            R = np.sqrt((x - nucleus.x)**2 + (y - nucleus.y)**2 + (z - nucleus.z)**2)
            nuclear_attraction -= (nucleus.charge / R) * ao_b.phi
        return nuclear_attraction

    # why is the time module failed
    h_time_start = time.time()

    # laplacian calculation from filters makes bad assumptions about gird size; / GRID**2 corrects this
    del2 = sci_fil.laplace(ao_b.phi) / GRID ** 2
    V = calculate_nuclear_attraction(ao_b, nuclei)
    h_function = ao_a.phi * ((-0.5)*del2 + V)
    h = integrate(h_function)

    h_time_end = time.time()
    if VERBOSE:
        name = f'<{ao_a.name}|h|{ao_b.name}>'
        total_time = h_time_end - h_time_start
        log.append(f'h element {name}, time = {total_time}, value = {h}')

    return h


def calculate_ab_g_cd(ao_a, ao_b, ao_c, ao_d):
    """all aos are orbital objects.
    This function returns an element of the G tensor as a float
    This function is intended to be used for both coulomb and exchange"""

    def a1(i, j, k):
        return ao_a.phi[i][j][k]

    def b1(i, j, k):
        return ao_b.phi[i][j][k]

    # should be unneeded, but I'll leave these in for now.
    def c1(i, j, k):
        return ao_c.phi[i][j][k]

    def d1(i, j, k):
        return ao_d.phi[i][j][k]

    name = f'<{ao_a.name},{ao_b.name}|g|{ao_c.name},{ao_d.name}>'
    g_time_start = time.time()

    if VERBOSE:
        print(f'starting integration of {name}')
        print('this might take a while')
        g_loading_bar = LoadingBar((len(x) - 2) ** 2)

    three_d_cross_section = np.zeros_like(x)

    # traverses indices, not xyz
    for i in range(1, len(x) - 1):
        for j in range(1, len(y) - 1):
            for k in range(1, len(z) - 1):
                # convert from index to value
                a = x_range[i]
                b = y_range[j]
                c = z_range[k]
                # added GRID / 2 to account for /0 error when taking 1/r
                # As GRID -> 0, this error also -> 0
                # if one thinks of the volume defined by 8 closest grid points, adding GRID/2 to the radius does not
                # simply extends the point being evaluated from the corner of this cube closer to the center
                r = find_distance(x, y, z, a, b, c) + GRID / 2
                three_d_cross_section[i, j, k] = integrate(a1(i, j, k) * b1(i, j, k) * ao_c.phi * ao_d.phi / r)
            if VERBOSE:
                g_loading_bar.update(1)

    g = integrate(three_d_cross_section)

    g_time_end = time.time()
    if VERBOSE:
        total_time = g_time_end - g_time_start
        log.append(f'g element {name}, time = {total_time}, value = {g}')

    return g


def find_nearest_gridpoint(x0, y0, z0, gridspace, type='value'):
    x_val, y_val, z_val = gridspace[0], gridspace[1], gridspace[2]

    i = np.argmin(abs(x_val - x0))
    j = np.argmin(abs(y_val - y0))
    k = np.argmin(abs(z_val - z0))

    if type == 'value':
        return x_val[i], y_val[j], z_val[k]
    elif type == 'index':
        return i, j, k
    else:
        print('Error! invalid argument for find_nearest_gridpoint().')
        exit()


def make_h_matrix(ao_list, nucleii):
    n = len(ao_list)
    h_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            h_matrix[i][j] = calculate_a_h_b(ao_list[i], ao_list[j], nucleii)

    return h_matrix


def make_g_tensor(ao_list):
    n = len(ao_list)
    g_tensor = np.full((n, n, n, n), -1)

    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    if g_tensor[a][b][c][d] == -1:
                        g_tensor[a][b][c][d] = g_tensor[a][b][d][c] = g_tensor[b][a][c][d] = g_tensor[b][a][d][c] =\
                            g_tensor[c][d][a][b] = g_tensor[c][d][b][a] = g_tensor[d][c][a][b] = g_tensor[d][c][b][a] =\
                            calculate_ab_g_cd(ao_list[a], ao_list[b], ao_list[c], ao_list[d])

    return g_tensor


def do_RHF():
    # set everything up
    nucleii = create_atoms(xyz_data)
    atomic_orbitals = create_aos(nucleii)
    for ao in atomic_orbitals:
        ao.normalize()

    h_matrix = make_h_matrix(atomic_orbitals, nucleii)
    g_tensor = make_g_tensor(atomic_orbitals)

    print(h_matrix)
    print(g_tensor)

    log.append(f"""
ao vector:
{[ao.name for ao in atomic_orbitals]}

h_matrix:
{h_matrix}

g_tensor:
{g_tensor}
""")

    print("RHF is currently unfinished!")

#     log.append(f"""
# summary:
# nuclear repulsion = {nuclear_repulsion}
# total one e energy = {total_onee}
# total coulomb = {total_coulomb}
# total energy = {total_energy}
# """)

    return 0


def do_UHF():
    print("Error: UHF currently unsupported")
    return 0


if __name__ == '__main__':
    total_time_start = time.time()
    log = []
    # default values
    SIZE = 3
    GRID = 0.04
    VERBOSE = False
    PLOTS = False

    commands, xyz_data, charge, mult, mo_list, coulomb_cs = read_file(sys.argv[1])

    log.append(f"""commands: {commands}
xyz data: {xyz_data}
charge, multiplicity: {charge}, {mult}
mos: {mo_list}
coulomb cross sections: {coulomb_cs}""")

    # assign commands
    for instruction in commands:
        if instruction[0] == 'size':
            SIZE = float(instruction[1])
        elif instruction[0] == 'grid':
            GRID = float(instruction[1])
        elif instruction[0] == 'verbose':
            VERBOSE = True
        elif instruction[0] == 'plots':
            PLOTS = True
        else:
            print(f'Error! Unrecognized argument {instruction[0]}')

    # Create a 3 dimensional gridspace
    # I want to make this a function, but every single variable might need to be global
    # most already are
    grid_time_start = time.time()

    # Setting up the electron grid
    # This guarantees that the center two grid points are +/- GRID/2 and that the grid extends to or just past SIZE
    # Note that other integer points may be on the grid - only (0,0,0) is guaranteed to be off grid.
    e_end = GRID * (round(SIZE / GRID) + 1/2)
    x_range = np.arange(-e_end, e_end + GRID, GRID)
    y_range = np.arange(-e_end, e_end + GRID, GRID)
    z_range = np.arange(-e_end, e_end + GRID, GRID)

    electron_gridspace = [x_range, y_range, z_range]
    x, y, z = np.meshgrid(x_range, y_range, z_range)

    # setting up the nuclear grid
    # the nuclear grid is half a grid unit off from the electron grid to avoid cusps
    # This guarantees that the center grid point is (0,0,0)
    N_end = GRID * round(SIZE / GRID)
    Nx_range = np.arange(-N_end, N_end + GRID, GRID)
    Ny_range = np.arange(-N_end, N_end + GRID, GRID)
    Nz_range = np.arange(-N_end, N_end + GRID, GRID)

    nuclear_gridspace = [Nx_range, Ny_range, Nz_range]
    Nx, Ny, Nz = np.meshgrid(Nx_range, Ny_range, Nz_range)

    grid_time_end = time.time()
    if VERBOSE:
        log.append(f'set up a grid with size = {SIZE} and res = {GRID}')
        log.append(f'grid_time = {grid_time_end - grid_time_start}')

    n_electrons = 0
    for row in xyz_data:
        n_electrons += periodic_table[row[0]]
    n_electrons -= charge
    if VERBOSE:
        log.append(f'# electrons = {n_electrons}')
    if (n_electrons % 2 == 0 and mult % 2 == 0) or (n_electrons % 2 != 0 and mult % 2 != 0):
        error_string = f'Error! n_electrons={n_electrons} and multiplicity={mult} not compatible! Exiting calculation.'
        print(error_string)
        log.append(error_string)
        exit()

    if mult == 1:
        energy = do_RHF()
    else:
        energy = do_UHF()

    print(f'final energy = {energy}')

    total_time_end = time.time()
    log.append(f'total time = {total_time_end - total_time_start}')
    with open('outfile.log', 'w') as out:
        out.writelines('\n'.join(log))
