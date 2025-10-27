from read_File import ReadFile
import numpy as np
import matplotlib.pyplot as plt
import math

#from scipy.io import npfile

from pathlib import Path
# from tqdm import tqdm # to show progress bar
from colorama import Fore, Style, init # to show a colourful end message
# import shutil
import cartopy.crs as ccrs
import imageio.v2 as imageio
# import os

class GeneratePlots:
    def __init__(self, reader: ReadFile):
        self.reader = reader

    def prepare_directory(prop_name, plane, dir_name = None):
        dir_name = f"./frames{plane}/frames{plane}_{prop_name}"

        dir_path = Path(dir_name)
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_name


    # Generates the Equatorial Plot for the property
    def generate_equatorial_plot(self, prop_type, prop_name, dir_path="."):
        ntheta = self.reader.ntheta
        nphi = self.reader.nphi

        radius = self.reader.radius
        phi = np.linspace(0., 2.*np.pi, nphi)

        mid_theta = ntheta // 2
        v_part = prop_type[:, mid_theta, :] # v = (phi,theta,r)
        v = np.tile(v_part, (self.reader.minc, 1))

        if v.shape[0] != len(phi):
            phi = np.linspace(0., 2*np.pi, v.shape[0])

        v_norm = v / max(abs(v.max()), abs(v.min())) #- (0.5 if prop_name == "entropy" else 0.0) #(v - v.min()) / (v.max() - v.min())

        rr, pphi = np.meshgrid(radius, phi, indexing='xy')
        xx = rr * np.cos(pphi)
        yy = rr * np.sin(pphi)

            # Plot filled contour
        plt.figure(figsize=(10.08, 10.08)) # to ensure that the plots are compatible with the movie
        contour = plt.contourf(xx, yy, v_norm, levels=100, cmap='magma' if prop_name=="entropy" else 'seismic', antialiased=True,
                                vmin=0 if prop_name=="entropy" else -1, vmax=1) # RdBu_r used Red for +ve and Blue for -ve
        plt.colorbar(contour, label=f'Normalized {prop_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Equatorial Plane: Normalised {prop_name}')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{dir_path}/frame_{self.number}.png", dpi=300)
        plt.close()


    # Generates the Meridonial Plot for the property
    def generate_meridional_plot(self, prop_type, prop_name, dir_path="."):
        ntheta = self.reader.ntheta

        radius = self.reader.radius
        theta = np.linspace(0., np.pi, ntheta)

        mid_phi = 0
        v = prop_type[mid_phi, :, ::-1]  # v = (phi,theta,r)


        if v.shape[0] != len(theta):
            theta = np.linspace(0., np.pi, v.shape[0])


        v_norm = v / max(abs(v.max()), abs(v.min())) - (0.5 if prop_name == "entropy" else 0.0) #(v - v.min()) / (v.max() - v.min())


        rr, ttheta = np.meshgrid(radius, theta, indexing='xy')
        zz = rr * np.cos(ttheta)
        xx = rr * np.sin(ttheta)

        # Plot filled contour
        plt.figure(figsize=(6.08, 6.08)) # to ensure that the image size is compatible with the movie
        contour = plt.contourf(xx, zz, v_norm, levels=100, cmap='seismic', antialiased=True, vmin=-1, vmax=1) # RdBu_r uses Red for +ve and Blue for -ve
        plt.colorbar(contour, label=f'Normalized {prop_name}')
        plt.xlabel('x')
        plt.ylabel('z')
        plt.title(f'Meridional Plane: Normalised {prop_name}')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{dir_path}/frame_{self.number}.png")
        plt.close()


    # Generates .png files of the plots of the requested property
    def generate_plot(self, srcDir, prop, plot_type):

        valid_props = {"vr", "vtheta", "vphi", "entropy"}
        if prop not in valid_props:
            raise ValueError(f"[ERROR] Invalid property '{prop}'. Must be one of {valid_props}")

        valid_plots = {"Eq", "Mer", "Mol", "Ortho"}
        if plot_type not in valid_plots:
            raise ValueError(f"[ERROR] Invalid plane '{plot_type}'. Must be one of {valid_plots}")

        # Creates a directory to store the plot images in, and returns the name of the directory
        targetDir = self.prepare_directory(prop, plot_type)

        if prop == "vr":
            data_to_plot = self.reader.vr
        elif prop == "vtheta":
            data_to_plot = self.reader.vtheta
        elif prop == "vphi":
            data_to_plot = self.reader.vphi
        elif prop == "entropy":
            data_to_plot = self.reader.entropy


        if plot_type == "Mer":
            self.generate_meridional_plot(data_to_plot, prop, targetDir)
        elif plot_type == "Eq":
            self.generate_equatorial_plot(data_to_plot, prop, targetDir)
        elif plot_type == "Mol":
            self.generate_Mollweide_Plot(data_to_plot, prop, targetDir)
        elif plot_type == "Ortho":
            self.generate_Ortho_Plot(data_to_plot, prop, targetDir)

        print(Fore.GREEN + "\n ... Task Completed" + Style.RESET_ALL)

    # Creates a movie
    def create_movie(src_dir, dest_dir, prop_name, plot_name):
        image_list = []
        dir_path = Path(src_dir)
        non_png_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() != ".png"]

        # Check for non-PNG files
        if non_png_files:
            raise ValueError(f"Non-PNG files found: {[f.name for f in non_png_files]}")

        # Only read .png files
        for f in sorted(dir_path.iterdir(), key=lambda f: int(''.join(filter(str.isdigit, f.stem)))):
            if f.is_file() and f.suffix.lower() == ".png":
                image_list.append(imageio.imread(f))

        # Save movie
        base_filename = f"movie{plot_name}_{prop_name}"
        movie_path = Path(f"{dest_dir}/{base_filename}.mp4")

        # Checks if movie of same name already exists. If it does, then creates movie with new name
        count = 1
        while movie_path.exists():
            movie_path = Path(f"{dest_dir}/{base_filename}({count}).mp4")
            count += 1

        imageio.mimsave(movie_path, image_list, fps=3)

        print(Fore.GREEN + "Movie Created" + Style.RESET_ALL)


    # To generate .png of the Mollweide projection of the data
    def generate_Mollweide_Plot(self, prop_type, prop_name, dir_path="."):
        nr = self.reader.nr # is this really required?
        ntheta = self.reader.ntheta
        nphi = self.reader.nphi

        theta = np.linspace(0., np.pi, ntheta)  # colatitude [0, pi]

        phi_part = np.linspace(0., 2*np.pi, nphi-1)  # partial azimuthal angle

        v = prop_type[:, :, 0]  # shape (phi_part, theta) , 0 corresponds to outermost radius

        # Tile phi and data to span full azimuth (0 to 2pi)
        phi = np.tile(phi_part, 1)
        v_full = np.tile(v, (self.reader.minc, 1))  # shape becomes (phi, theta)

        # Create meshgrid
        theta_grid, phi_grid = np.meshgrid(theta, phi)  # both shape (phi, theta)

        # Convert (theta, phi) to (latitude, longitude) in radians
        lon = phi_grid - np.pi           # Convert longitude from [0, 2*pi] to [-pi, pi]
        lat = 0.5*np.pi - theta_grid     # Convert colatitude from [0, pi] to [−pi/2, pi/2]

        v_norm = v_full / max(abs(v_full.max()), abs(v_full.min()))

        # Plot
        fig = plt.figure(figsize=(10.08, 6.08)) # to ensure compatibility with movie making
        ax = fig.add_subplot(111, projection='mollweide')

        pc = ax.pcolormesh(lon, lat, v_norm, shading='auto', cmap='seismic', vmin = -1, vmax = 1)

        # Set ticks
        xticks_deg = np.arange(-150, 180, 30)
        ax.set_xticks(np.radians(xticks_deg))
        ax.set_xticklabels([f'{deg}°' for deg in xticks_deg])
        ax.grid(True)

        # Colorbar
        cbar = plt.colorbar(pc, ax=ax, orientation='horizontal', pad=0.07)
        cbar.set_label(f'{prop_name} (normalized)')

        plt.title(f'{prop_name} on Mollweide Projection')
        plt.tight_layout()
        plt.savefig(f"{dir_path}/frame_{self.reader.number}.png", dpi=300)
        plt.close()

        # To generate .png files of orthographic projection of the spherical surface
    def generate_Ortho_Plot(self, prop_type, prop_name, dir_path="."):
        nr = self.reader.nr
        ntheta = self.reader.ntheta
        nphi = self.reader.nphi

        # Angular coordinates
        theta = np.linspace(0., np.pi, ntheta)               # colatitude [0, pi]
        phi_part = np.linspace(0., 2*np.pi, nphi-1)  # azimuthal angle for partial wedge

        v = prop_type[:, :, 0]                         # shape (phi_part, theta) , 0 corresponds to outermost radius

        # Tile phi and data to complete full 2π in azimuth
        phi = np.tile(phi_part, 1)                        # full azimuth
        v_full = np.tile(v, (self.reader.minc, 1))                    # shape becomes (phi, theta)

        # Create meshgrid in (lon, lat)
        lon_grid, lat_grid = np.meshgrid(phi, theta, indexing='ij')  # shape (phi, theta)

        # Convert to lat/lon
        lon = (lon_grid * 180 / np.pi) - 180                        # longitudes: -180 to 180
        lat = 90 - (lat_grid * 180 / np.pi)                         # latitudes: 90 (north pole) to -90 (south)

        v_full_norm = v_full #/ max(abs(v_full.min()), abs(v_full.max()))
        vmax = max(abs(v_full.max()), abs(v_full.min()))
        vmin = -vmax

        # vmin = np.percentile(v_full, 2)
        # vmax = np.percentile(v_full, 98)

        # Plot with Cartopy
        fig = plt.figure(figsize=(10.08, 10.08))

        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=10))

        contour = ax.pcolormesh(lon, lat, v_full_norm, shading='gouraud', transform=ccrs.PlateCarree(), cmap="seismic", vmin = vmin, vmax = vmax, zorder=1)

        # ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='black')

        # To plot the tangent cylinder's intersection with the outer surface
        tangent_lat = 90 - math.degrees(math.acos(0.75))
        tang_lons = np.linspace(-180, 180, 500)
        tang_lats = np.full_like(tang_lons, tangent_lat)
        ax.plot(tang_lons, tang_lats, color='black', linewidth=1.5, linestyle=':', transform=ccrs.Geodetic(), zorder=2)
        ax.text(0, tangent_lat + 3, "Tangent Cylinder", color='black', fontsize=12, ha='center', transform=ccrs.Geodetic())


        import matplotlib.ticker as mticker
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle=':', color='gray', zorder=3)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
        gl.top_labels = False
        gl.right_labels = False

        plt.title(f"Orthographic Projection of {prop_name}", fontsize=14)
        cbar = plt.colorbar(contour, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label(prop_name)

        plt.savefig(f"{dir_path}/frame_{self.reader.number}.png", bbox_inches='tight', dpi=300)
        plt.close()