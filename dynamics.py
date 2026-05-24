import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- 1. Problem Parameters (from the image) ---
M = 15.0       # Mass of rod (kg)
L = 0.6        # Length of rod (m)
P = 200.0      # Pulling force (N)
G = 9.81       # Acceleration due to gravity (m/s^2)

# Geometry constants
ALPHA = np.deg2rad(45)  # Incline angle (45 degrees)
INA = np.sin(ALPHA)
COA = np.cos(ALPHA)

# Moment of Inertia for a rod about its center of mass
IG = (1/12) * M * L**2

# --- 2. Physics Engine: Equation of Motion ---
# We use the Lagrangian method (T - V) to derive the EOM: J(theta)*ddot(theta) + C(theta,dot_theta) = Q(theta)
# The state vector 'y' is [theta, angular_velocity]

def rod_dynamics(t, y):
    theta, omega = y

    # Prevent potential singularity right at theta = 0, though start from rest should be fine
    theta = max(theta, 1e-6)

    # Simplified common geometric terms
    sint = np.sin(theta)
    cost = np.cos(theta)
    sin_sum = np.sin(theta + ALPHA)
    cos_sum = np.cos(theta + ALPHA)
    
    # 2a. Mass/Inertia Terms (coefficients of ddot_theta)
    term_da = sin_sum**2 / INA**2
    term_d = (L / sin_sum)**2 * sint * (COA*sin_sum - sin_sum) # geometric velocity mapping term

    # Mass matrix term (Effective rotational inertia J(theta))
    J_theta = IG + (M/4) * term_da + (M/INA**2) * sint**2

    # 2b. Centripetal/Coriolis Terms (coefficients of omega^2)
    # This involves partial derivatives, simplified for implementation
    # A simplified approximation is used for this simulation's clarity
    C_term = (M * L**2 / (8 * INA**2)) * omega**2 * np.sin(2*theta + ALPHA)

    # 2c. Force/Generalized Force Terms
    # Work done by force P on point B
    Q_P = P * L * cost
    # Work done by gravity on center of mass G (downward)
    Q_g = -M * G * (L/2) * sint # Positive because angle is from horizontal

    # Full Force vector Q(theta)
    Q_theta = Q_P + Q_g - C_term

    # Solve for angular acceleration alpha = ddot(theta)
    alpha = Q_theta / J_theta
    return [omega, alpha]

# --- 3. Initial Conditions and Simulation Time ---
# theta0 = 0.0, omega0 = 0.0 (rest)
y0 = [0.0, 0.0]
# Simulate for 1.5 seconds, which should capture the swing past 45 deg
t_span = (0, 1.5)
t_eval = np.linspace(t_span[0], t_span[1], 300) # 300 frames for smooth animation

# Solve the differential equation
print("Simulating...")
solution = solve_ivp(rod_dynamics, t_span, y0, t_eval=t_eval, rtol=1e-6)
print("Simulation complete.")

# --- 4. Post-Processing for Animation ---
times = solution.t
thetas = solution.y[0]

# Calculate Cartesian positions of key points at each time step
xA_points = (L * np.sin(thetas) / INA) * COA
yA_points = (L * np.sin(thetas) / INA) * INA # simplified yA = L sin(theta)

xB_points = L * np.cos(thetas)
yB_points = np.zeros_like(xB_points)

# Position of Center of Mass (G)
xG_points = (xA_points + xB_points) / 2
yG_points = (yA_points + yB_points) / 2

# Find the frame index just before theta reaches 45 degrees (pi/4)
target_theta = np.pi / 4
target_idx = np.abs(thetas - target_theta).argmin()
calculated_omega = solution.y[1][target_idx]
calculated_theta = solution.y[0][target_idx] * 180 / np.pi
print(f"Target theta reached: {calculated_theta:.2f}° (expected ~45°)")
print(f"Simulated Angular Velocity (omega): {calculated_omega:.3f} rad/s")

# --- 5. Matplotlib Animation ---
fig, ax = plt.subplots(figsize=(8, 6))
# Set crucial aspect ratio to 'equal' so dimensions are accurate
ax.set_aspect('equal')
ax.set_xlim(-0.2, 0.8)
ax.set_ylim(-0.1, 0.8)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title(f'Dynamics of Rod System (P={P}N, m={M}kg)')
ax.grid(True, linestyle='--')

# 5a. Static Track and Force Arrow
# Draw the tracks
track_bend_x = -0.1
incline_x = np.linspace(-0.2, track_bend_x, 100)
incline_y = (incline_x - track_bend_x) * -INA / -COA + 0.0 # simple slope
track_incline, = ax.plot(incline_x, incline_y, 'gray', lw=2)
track_horiz, = ax.plot([track_bend_x, 0.8], [0, 0], 'gray', lw=2)

# Drawing force P arrow at point B (final resting position approx)
force_start_x = 0.6
force_start_y = 0.0
force_end_x = force_start_x + 0.1
ax.annotate("", xy=(force_start_x, force_start_y), xytext=(force_end_x, force_start_y),
            arrowprops=dict(facecolor='red', shrink=0, width=2, headwidth=8))
ax.text(force_end_x + 0.02, force_start_y - 0.03, f'P = {P} N', color='red')

# 5b. Dynamic Elements (to be updated)
# Draw the rod
rod_line, = ax.plot([], [], 'o-', lw=6, markersize=12, color='peru', mfc='black', mec='black')

# Draw the path of the Center of Mass (G)
g_path, = ax.plot([], [], ':', lw=1.5, color='blue', alpha=0.6)

# Display real-time simulation data
text_template = 'Time: {:.2f}s\nAngle: {:.1f}°\nAngular Vel: {:.3f} rad/s'
data_text = ax.text(0.5, 0.65, '', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

# Draw a specific marker when theta = 45 is reached
hit_marker, = ax.plot([], [], 'ro', markersize=20, mfc='none', mew=2, label=f'@ {target_theta*180/np.pi:.1f}°')
# ax.legend()

# 5c. The Initialization Function
def init():
    rod_line.set_data([], [])
    g_path.set_data([], [])
    hit_marker.set_data([], [])
    data_text.set_text('')
    return rod_line, g_path, hit_marker, data_text

# 5d. The Animation Frame Update Function
def animate(i):
    # Get current positions
    this_xA = xA_points[i]
    this_yA = yA_points[i]
    this_xB = xB_points[i]
    this_yB = yB_points[i]

    # Update the rod position
    rod_line.set_data([this_xA, this_xB], [this_yA, this_yB])

    # Update the center of mass path
    g_path.set_data(xG_points[:i], yG_points[:i])

    # Highlight target moment
    if i == target_idx:
        hit_marker.set_data([this_xA], [this_yA])
    elif i < target_idx or i > target_idx + 10: # hide quickly
        hit_marker.set_data([],[])

    # Update text data
    current_time = times[i]
    current_theta_deg = thetas[i] * 180 / np.pi
    current_omega = solution.y[1][i]
    data_text.set_text(text_template.format(current_time, current_theta_deg, current_omega))

    return rod_line, g_path, hit_marker, data_text

# 5e. Run the Animation
print("Creating animation (takes a moment)...")
# Run at 60fps equivalent to 1x speed
anim = FuncAnimation(fig, animate, init_func=init, frames=len(times), interval=1000/(300/1.5), blit=True)
print("Opening viewer...")
plt.show()

# Optional: Un-comment the next lines to save the animation as a video file (requires FFMpeg)
# print("Saving to video file...")
# writer = FFMpegWriter(fps=30)
# anim.save("rod_animation.mp4", writer=writer)
# print("Animation saved!")