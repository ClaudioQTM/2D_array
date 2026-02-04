
#%% Calculation the self energy at the center of 1st BZ. Compare the plot to Fig. 3 in E. Shahmoon et al 2017
n_points = 20
start_point = 0.12
end_point = 0.999
self_energy_center_bz = Parallel(n_jobs = 10)(delayed(self_energy)(0,0, aa*2*np.pi,square_lattice.d,1,1,alpha) for aa in np.linspace(start_point, end_point, n_points))



# Convert to numpy array and extract real/imaginary parts
self_energy_array = np.array(self_energy_center_bz)
norm_self_energy_array = self_energy_array/float(square_lattice.gamma)
real_part = norm_self_energy_array.real
imag_part = norm_self_energy_array.imag


real_part = -real_part
imag_part = -imag_part
# Create x-axis from the aa values
aa_values = np.linspace(start_point, end_point, n_points)


# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot real part
ax1.plot(aa_values, real_part, 'b-o', linewidth=2, markersize=6, label='Real part')
ax1.set_xlabel(r'$a$ (lattice spacing)', fontsize=12)
ax1.set_ylabel(r'Re($\sigma$)', fontsize=12)
ax1.set_title('Real Part of Self-Energy at Center of 1st BZ', fontsize=14, fontweight='bold')
ax1.set_ylim(-3, 1)
ax1.set_xlim(0, 1)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot imaginary part
ax2.plot(aa_values, imag_part, 'r-o', linewidth=2, markersize=6, label='Imaginary part')
ax2.set_xlabel(r'$a$ (lattice spacing)', fontsize=12)
ax2.set_ylabel(r'Im($\sigma$)', fontsize=12)
ax2.set_title('Imaginary Part of Self-Energy at Center of 1st BZ', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11)

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print("SELF-ENERGY STATISTICS")
print("="*60)
print(f"Real part - Min: {real_part.min():.6e}, Max: {real_part.max():.6e}, Mean: {real_part.mean():.6e}")
print(f"Imaginary part - Min: {imag_part.min():.6e}, Max: {imag_part.max():.6e}, Mean: {imag_part.mean():.6e}")
print(f"Magnitude - Min: {np.abs(self_energy_array).min():.6e}, Max: {np.abs(self_energy_array).max():.6e}")
print("="*60 + "\n")

#%%