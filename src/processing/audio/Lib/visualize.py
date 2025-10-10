import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def plot_spectrogram_from_spectrogram(Y, t, f, fmax=None, gain=0, range_spec=80, DPI=300, eps = 1e-12, filename=None):
    """
    Plot spectrogram

    Args:
        Y (np.ndarray):
            The time-frequency representation [(nb_of_channels), nb_of_frames, nb_of_bins].
        t (np.ndarray):
            Time vector [nb_of_frames]
        f (np.ndarray):
            Frequency vector [nb_of_bins,1]
        fmax:
            Max frequency to display in spectrogram
        gain:
        range:
        filename (str):
            Filename for figure. If filename=None, it wont save.
    """
    if Y.ndim == 3:
        Y = Y[0,:,:]
    if fmax == None:
        idfmax = f.shape[0]
    else:
        idfmax = int(fmax/f[-1]*f.shape[0])

    fig, ax = plt.subplots(figsize=(4,4/3.2))
    data = 20 * np.log10((np.abs(Y[:, :idfmax]) + eps) / (np.max(np.abs(Y[:, :idfmax])) + eps)).T
    ax.pcolormesh(t, f[:idfmax]/1000, data, vmin=-gain-range_spec, vmax=-gain, shading='gouraud', cmap="magma", rasterized=True)
    ax.set_ylabel('Frequency [kHz]', fontsize=8)
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=6)
    # fig.tight_layout()
    
    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'.pdf',bbox_inches='tight', pad_inches=0.01, dpi=DPI)
        plt.savefig(filename+'.png',bbox_inches='tight', pad_inches=0.01, dpi=DPI)



def hist_direction_from_DOAs(DOAs, filename = None):
    # For clap
    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    y, x, _ = ax.hist(DOAs[:,0], 36)
    max_hits = np.max(y)
    ax.plot((15,15),(0,max_hits*2), '--r')
    ax.plot((75,75),(0,max_hits*2), ':r')
    ax.plot((133,133),(0,max_hits*2), ':r')
    ax.plot((213,213),(0,max_hits*2), ':r')
    ax.plot((343,343),(0,max_hits*2), ':r')
    ax.set_ylim(0,max_hits+10)
    ax.set_xlabel(r"$\Theta$ [deg]", fontsize=8)
    ax.set_ylabel("Number of hits", fontsize=8)
    ax.set_xticks(np.arange(0, 361, 60))
    ax.set_yticks(np.arange(0,np.max(y), 100))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'_theta.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_theta.png',bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    y, x, _ = ax.hist(DOAs[:,1], 18)
    max_hits = np.max(y)
    ax.plot((42,42),(0,max_hits*2), ':r')
    ax.plot((175,175),(0,max_hits*2), '--r')
    ax.set_ylim(0,max_hits+10)
    ax.set_xlabel(r"$\Phi$ [deg]", fontsize=8)
    ax.set_ylabel("Number of hits", fontsize=8)
    ax.set_xticks(np.arange(0, 181, 60))
    ax.set_yticks(np.arange(0,np.max(y), 100))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'_phi.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)

def plot_SRP_from_SRP(SRP, frame_ids, max_marker = False, filename = None):
    nTheta = SRP.shape[0]
    nPhi = SRP.shape[1]

    if type(frame_ids) == int:
        frame_ids = np.array([frame_ids])

    for frame_id in frame_ids:
        fig, ax = plt.subplots(figsize=(4.8/2,4.8/2/2))
        ax.pcolormesh(SRP[:,:,int(frame_id)].T, shading='gouraud', rasterized=True)
        ax.set_xlabel(r"$\Theta$ [deg]", fontsize=8, labelpad=2)
        ax.set_ylabel(r"$\Phi$ [deg]", fontsize=8, labelpad=1)
        ax.set_xticks(np.arange(0, 361, 90))
        ax.set_yticks(np.arange(0, 181, 90))
        ax.tick_params(axis='both', which='major', labelsize=6)
        # fig.subplots_adjust(bottom=0.15)

        if max_marker:
            SRP_argmax = np.unravel_index(np.argmax(SRP[:,:,frame_id]), (nTheta, nPhi))
            ax.plot(SRP_argmax[0], SRP_argmax[1], 'xr', markersize=6, mfc='none', markeredgewidth=1)

        if filename == None:
            plt.pause(1) 
        else:
            # plt.savefig(filename+'_frame'+str(int(frame_id))+'.svg',bbox_inches='tight')
            plt.savefig(filename+'_frame'+str(int(frame_id))+'.pdf',bbox_inches='tight',pad_inches=0.01, dpi=300)


def plot_DOAs_single_source(DOA_measured, t, filename=None):

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t,DOA_measured[:,0],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(r"$\Theta$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 361, 60))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_theta.svg',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_theta.png',bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t,DOA_measured[:,1],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(r"$\Phi$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 181, 60))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_phi.svg',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)


def plot_DOAs_multiple_methods(doa_data, t, method_names, filename=None):
    """
    Plots theta and phi angles over time for different localization methods using LaTeX-rendered labels.

    Args:
        doa_data (list of np.ndarray): 
            List of arrays containing DOA data for each method. Each array should have shape (nb_of_frames, 2) with columns [theta, phi].
        t (np.ndarray):
            Time vector.
        method_names (list of str): 
            List of method names corresponding to each DOA data array.
        filename (str, optional):
            Filename to save the plot. If None, the plot is displayed with a title.
    """
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Plot theta over time
    for data, name in zip(doa_data, method_names):
        axs[0].plot(t, data[:, 0], 'o', markersize=8, markeredgewidth=1, label=fr'{name} $\theta$') 
    axs[0].set_ylabel(r'$\theta$ [deg]')
    axs[0].set_yticks(np.arange(0, 361, 60))
    axs[0].legend()
    axs[0].grid(True)

    # Plot phi over time
    for data, name in zip(doa_data, method_names):
        axs[1].plot(t, data[:, 1], 'o', markersize=8, markeredgewidth=1, label=fr'{name} $\phi$')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel(r'$\phi$ [deg]')
    axs[1].set_yticks(np.arange(0, 181, 60))
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    if filename is None:
        # Add suptitle with adjusted vertical position
        fig.suptitle('DOA Angles Over Time for Different Methods')
        # Adjust subplot parameters to give space for the suptitle
        fig.subplots_adjust(top=0.95)
        plt.show()
    else:
        plt.savefig(f'{filename}_doas.svg', bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)


def plot_DOAs_errors(doa_error_data, t, method_names, filename=None):
    """
    Plots theta and phi angles over time for different localization methods using LaTeX-rendered labels.

    Args:
        doa_error_data (list of np.ndarray): 
            List of arrays containing DOA data for each method. Each array should have shape (nb_of_frames, 2) with columns [theta, phi].
        t (np.ndarray):
            Time vector.
        method_names (list of str): 
            List of method names corresponding to each DOA data array.
        filename (str, optional):
            Filename to save the plot. If None, the plot is displayed with a title.
    """
    fig, ax = plt.subplots(figsize=(12, 6), sharex=True)

    # Plot theta over time
    for error, name in zip(doa_error_data, method_names):
        ax.plot(t, error, 'o', markersize=15, markeredgewidth=10, label=fr'{name} Error') 
    ax.set_ylabel(r'Absolute error [deg]')
    ax.set_yticks(np.arange(0, 31, 5))
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    if filename is None:
        # Add suptitle with adjusted vertical position
        fig.suptitle('DOA Error Over Time for Different Methods')
        # Adjust subplot parameters to give space for the suptitle
        fig.subplots_adjust(top=0.95)
        plt.show()
    else:
        plt.savefig(f'{filename}_doas_error.svg', bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)


def plot_DOAs_errors_log_scale(doa_error_data, t, method_names, pseudocount=1e-1, filename=None):
    """
    Plots the absolute DOA errors over time for different localization methods using a logarithmic scale.

    Args:
        doa_error_data (list of np.ndarray): 
            List of arrays containing DOA error data for each method.
            Each array should have shape (nb_of_frames,).
        t (np.ndarray):
            Time vector.
        method_names (list of str): 
            List of method names corresponding to each DOA error data array.
        filename (str, optional):
            Filename to save the plot. If None, the plot is displayed.
        pseudocount (float, optional):
            Small constant added to error values to avoid issues with logarithms of zero.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot DOA errors over time
    for error, name in zip(doa_error_data, method_names):
        # Ensure all errors are positive by adding a pseudocount
        error_with_pseudocount = np.abs(error) + pseudocount
        ax.plot(t, error_with_pseudocount, 'o', markersize=2, markeredgewidth=0.4, label=fr'{name} Error')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Absolute Error [deg]')
    ax.set_yscale('log')  # Set y-axis to logarithmic scale
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    if filename is None:
        fig.suptitle('DOA Error Over Time for Different Methods (Log Scale)')
        fig.subplots_adjust(top=0.95)
        plt.show()
    else:
        plt.savefig(f'{filename}_doas_error_log_scale.svg', bbox_inches='tight', pad_inches=0.01)
        plt.close(fig)

def plot_spherical(doa_data, method_names):
    """
    Plots DOA data on a spherical coordinate system using a 3D scatter plot.

    Args:
        doa_data (list of np.ndarray): List of arrays containing DOA data for each method.
                                        Each array should have shape (nb_of_frames, 2) with columns [theta, phi].
        method_names (list of str): List of method names corresponding to each DOA data array.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for data, name in zip(doa_data, method_names):
        theta = np.radians(data[:, 0])
        phi = np.radians(data[:, 1])
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        ax.scatter(x, y, z, label=name, s=10)

    # Create a wireframe sphere for reference
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.5, linewidth=0.5)

    ax.set_title('Spherical Plot of DOA')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def plot_grid_on_sphere(points):

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=10, color='blue')

    # Set plot title and labels
    ax.set_title('Fibonacci Sphere Distribution')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    plt.show()


def visualize_srp_theta_phi_heatmaps(SRP, spherical_grid, time, vmax=None):
    """
    Visualize SRP over time as two heatmaps: one for azimuth (theta) and one for elevation (phi).

    Args:
        SRP (np.ndarray): SRP values with shape [nb_of_frames, nb_of_doas].
        spherical_grid (np.ndarray): Spherical grid coordinates with shape [nb_of_doas, 3].
        time (np.ndarray): Time array with shape [nb_of_frames].
    """
    # Convert Cartesian coordinates to spherical angles
    x, y, z = spherical_grid[:, 0], spherical_grid[:, 1], spherical_grid[:, 2]
    theta = np.degrees(np.arctan2(y, x))  # Azimuth angle in degrees
    phi = np.degrees(np.arccos(z))        # Elevation angle in degrees

    # Ensure theta values are in the range [0, 360)
    theta = np.mod(theta, 360)

    # Sort indices for theta and phi
    theta_sort_idx = np.argsort(theta)
    phi_sort_idx = np.argsort(phi)

    # Sort angles
    theta_sorted = theta[theta_sort_idx]
    phi_sorted = phi[phi_sort_idx]

    # Sort SRP data accordingly
    SRP_theta_sorted = SRP[:, theta_sort_idx]
    SRP_phi_sorted = SRP[:, phi_sort_idx]

    # Create the plots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Azimuth (theta) heatmap
    im0 = axs[0].imshow(SRP_theta_sorted.T, aspect='auto', origin='lower',
                        extent=[time[0], time[-1], theta_sorted[0], theta_sorted[-1]],
                        cmap='viridis', vmax=vmax)
    axs[0].set_ylabel('Azimuth θ (degrees)')
    axs[0].set_title('SRP over Time and Azimuth')
    fig.colorbar(im0, ax=axs[0], label='SRP Intensity')

    # Elevation (phi) heatmap
    im1 = axs[1].imshow(SRP_phi_sorted.T, aspect='auto', origin='lower',
                        extent=[time[0], time[-1], phi_sorted[0], phi_sorted[-1]],
                        cmap='viridis', vmax=vmax)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Elevation φ (degrees)')
    axs[1].set_title('SRP over Time and Elevation')
    fig.colorbar(im1, ax=axs[1], label='SRP Intensity')

    plt.tight_layout()
    plt.show()

def plot_srp_theta_phi_heatmap_for_frames(SRP, spherical_grid, frame_ids, vmax=None, theta_res=1.0, phi_res=1.0):
    """
    Visualize SRP θ-φ heatmap for specific frames.

    Args:
        SRP (np.ndarray): [nb_of_frames, nb_of_doas]
        spherical_grid (np.ndarray): [nb_of_doas, 3] Cartesian coordinates
        frame_ids (list or array): Indices of frames to plot
        vmax (float, optional): Max value for color scale
    """
    # Convert Cartesian to spherical
    x, y, z = spherical_grid[:, 0], spherical_grid[:, 1], spherical_grid[:, 2]
    theta = np.degrees(np.arctan2(y, x)) % 360  # [0, 360)
    phi = np.degrees(np.arccos(z))              # [0, 180]

    # Create regular grid
    theta_grid = np.arange(0, 360 + theta_res, theta_res)
    phi_grid = np.arange(0, 180 + phi_res, phi_res)
    Theta, Phi = np.meshgrid(theta_grid, phi_grid)

    for frame_id in frame_ids:
        # Interpolate SRP values onto the regular grid
        srp_interp = griddata(
            (theta, phi),
            SRP[frame_id],
            (Theta, Phi),
            method='linear',
            fill_value=np.nan
        )

        plt.figure(figsize=(8, 4))
        plt.imshow(
            srp_interp,
            aspect='auto',
            origin='lower',
            extent=[theta_grid[0], theta_grid[-1], phi_grid[0], phi_grid[-1]],
            cmap='viridis',
            vmax=vmax
        )
        plt.xlabel('Azimuth θ [deg]')
        plt.ylabel('Elevation φ [deg]')
        plt.title(f'SRP θ-φ Heatmap (Frame {frame_id})')
        plt.colorbar(label='SRP Intensity')
        plt.tight_layout()
        plt.show()
        
def plot_direction_from_DOAs_with_noise(DOA_noise, DOA_measured, t, filename=None):

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t,DOA_noise[:,0],'vb', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.plot(t,DOA_measured[:,0],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(r"$\Theta$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 361, 60))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_theta.svg',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_theta.png',bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t,DOA_noise[:,1],'vb', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.plot(t,DOA_measured[:,1],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(r"$\Phi$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 181, 60))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_phi.svg',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)


def plot_direction_from_DOAs_with_target(DOA_target, DOA_measured, t, speech_mask, filename=None):

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t[speech_mask],DOA_target[speech_mask,0], 'xk', markersize=2, markeredgewidth=0.4)
    ax.plot(t[speech_mask],DOA_measured[speech_mask,0],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylabel(r"$\Theta$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 361, 60))
    ax.tick_params(axis='both', which='major', labelsize=6)
    

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_theta.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_theta.png',bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots(figsize=(4.8/2,4.8/2*(3/4)))
    ax.plot(t[speech_mask],DOA_target[speech_mask,1],'xk', markersize=2, markeredgewidth=0.4)
    ax.plot(t[speech_mask],DOA_measured[speech_mask,1],'og', markersize=2, markeredgewidth=0.4, mfc='none')
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.set_ylim(0, 90)
    ax.set_ylabel(r"$\Phi$ [deg]", fontsize=8)
    ax.set_yticks(np.arange(0, 91, 30))
    ax.tick_params(axis='both', which='major', labelsize=6)

    if filename == None:
        plt.pause(1) 
    else:
        plt.savefig(filename+'_phi.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)



def plot_dir_pat_2D(Rs, f, freq_id, filename=None):
    nTheta = Rs.shape[1]
    theta = np.linspace(0,360,nTheta)

    # 2D 0-180 deg plot : THETA
    fig, ax = plt.subplots(figsize=(4.8,4.8*(3/4)))
    ax.plot(theta[:nTheta//2+1],np.log10(Rs[freq_id,:nTheta//2+1]))
    ax.set_xlabel(r"$\Theta$ [deg]", fontsize=14)
    ax.set_ylabel(r"log10($\|D\|$)", fontsize=14)
    if filename == None:
        ax.set_title('Directivité, f = %i Hz' %f[freq_id])
    else:
        plt.savefig(filename+'_dirPlot_0-180_'+str(f[freq_id])+'Hz.pdf', bbox_inches='tight', pad_inches=0.01)


    # 2D polar plot : THETA
    fig = plt.figure(figsize=(4.8,4.8*(3/4)))
    ax = fig.add_subplot(111, projection='polar')
    ax.plot(theta*np.pi/180, np.log10(Rs[freq_id,:]))
    if filename == None:
        ax.set_title('Directivité, f = %i Hz' %f[freq_id])
    else:
        plt.savefig(filename+'_dirPlot_polar'+str(f[freq_id])+'Hz.pdf', bbox_inches='tight', pad_inches=0.01)


    # 2D-multifreq plot THETA
    F, Theta = np.meshgrid(f, theta, indexing='ij')

    # fig = plt.figure(figsize=(4.8/2,4.8/2*(3/4)))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(F,Theta, Rs, cmap=plt.cm.coolwarm)
    # plt.title('Directivité')
    # ax.set_xlabel('Freq [Hz]')
    # ax.set_ylabel('Theta [deg]')
    # ax.set_zlabel('|D|')
    # ax.set_zlim(0, 1)

    fig = plt.figure(figsize=(4.8,4.8*(3/4)))
    ax = fig.add_subplot(111)
    ax.pcolormesh(F/1000,Theta, np.log10(Rs), shading='gouraud', vmin=-2, rasterized=True)  # , vmax=1
    ax.set_xlabel('Frequency [kHz]', fontsize=14)
    ax.set_ylabel(r"$\Theta$ [deg]", fontsize=14)
    
    if filename == None:
        ax.set_title('|D|')
    else:
        plt.savefig(filename+'_dirPlot_multiFreq.pdf', bbox_inches='tight', pad_inches=0.01, dpi=300)


    plt.show()


def plot_dir_pat(Rs, scan_grid, f, filename = None):
    """
    Plot directivity pattern of 3D beamformer

    Args:
        Rs (np.ndarray):
            The beamformer directivity pattern in the frequency domain (from dir_pat()) [nb_of_bins_for_eval, nTheta, nPhi]
        scan_grid (np.ndarray):
            Spherical grid of radius = 1 [3, nTheta, nPhi]
        f (np.ndarray):
            frequencies for evaluation [nb_of_bins_for_eval,]
    """
    nTheta = Rs.shape[1]
    nPhi = Rs.shape[2]
    theta = np.linspace(0,360,nTheta)
    phi = np.linspace(0,180,nPhi)

    Rs_mean = np.mean(Rs, axis=0)

    # 3D directivity plot
    X = Rs_mean*scan_grid[0,:,:]
    Y = Rs_mean*scan_grid[1,:,:]
    Z = Rs_mean*scan_grid[2,:,:]


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(X,Y,Z, cmap=plt.cm.YlGnBu_r)
    # plt.title('Directivity averaged for all freq')
    ax.set_xlabel('$|H|(x)$', fontsize = 12)
    ax.set_ylabel('$|H|(y)$', fontsize = 12)
    ax.set_zlabel('$|H|(z)$', fontsize = 12)
    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'dir3D_av_freq.pdf',bbox_inches='tight',pad_inches=0.5)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)

    # # 2D polar plot : THETA
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='polar')
    # ax.plot(theta*np.pi/180, Rs_mean[:,nPhi//4])
    # plt.title('Directivity averaged for all freq for Theta when phi = %i deg' %phi[nPhi//4])
    # ax.set_xlabel('Theta')
    # ax.set_ylabel('|H|')

    # # 2D polar plot : PHI
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='polar')
    # ax.plot(phi*np.pi/180, Rs_mean[nTheta//8,:])
    # plt.title('Directivity averaged for all freq for Phi when theta = %i deg' %theta[nTheta//8])
    # ax.set_xlabel('Phi')
    # ax.set_ylabel('|H|')

    # 2D-multifreq plot THETA
    F, Theta = np.meshgrid(f, theta, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(F,Theta, Rs[:,:,nPhi//2], cmap=plt.cm.coolwarm)
    plt.title('Directivité, phi = %i deg' %phi[nPhi//4])
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('Theta [deg]')
    ax.set_zlabel('|H|')
    ax.set_zlim(0, 1)

    fig = plt.figure(figsize=(6,6*(3/4)))
    ax = fig.add_subplot(111)
    ax.pcolormesh(F/1000,Theta, Rs[:,:,nPhi//2], shading='gouraud')  # , vmax=1
    ax.set_xlabel('Frequency [kHz]', fontsize=14)
    ax.set_ylabel(r'$\Theta$ [deg]', fontsize = 14)
    # ax.set_title('$|H|$')
    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'dir2D_vs_freq_phi45d.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)

    # 2D-multifreq plot PHI
    F, Phi = np.meshgrid(f, phi, indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(F,Phi, Rs[:,nTheta//8,:], cmap=plt.cm.coolwarm)
    plt.title('Directivité, phi = %i deg' %theta[nTheta//2])
    ax.set_xlabel('Freq [Hz]')
    ax.set_ylabel('Phi [deg]')
    ax.set_zlabel('|H|')
    ax.set_zlim(0, 1)

    fig = plt.figure(figsize=(6,6*(3/4)))
    ax = fig.add_subplot(111)
    ax.pcolormesh(F/1000,Phi, Rs[:,nTheta//8,:], shading='gouraud')  # , vmax=1
    ax.set_xlabel('Frequency [kHz]', fontsize=14)
    ax.set_ylabel(r'$\Phi$ [deg]', fontsize = 14)
    # ax.set_title('|H|')

    if filename == None:
        plt.pause(1)
    else:
        plt.savefig(filename+'dir2D_vs_freq_theta45d.pdf',bbox_inches='tight',pad_inches=0.01)
        # plt.savefig(filename+'_phi.png',bbox_inches='tight', dpi=300)


    plt.show()