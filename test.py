import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev, interp1d

def generate_mock_data(num_points=30, img_size=1024, confidence_threshold=0.5):
    """Generate realistic trajectory points with some noise and random confidence values."""
    np.random.seed(42)  # For reproducibility
    
    # Generate a smooth curve (sinusoidal-like motion)
    x = np.linspace(100, img_size - 100, num_points)
    y = img_size // 2 + 100 * np.sin(np.linspace(0, 4 * np.pi, num_points)) + np.random.uniform(-10, 10, num_points)
    
    # Assign random confidence values (some low-confidence points)
    confidence = np.random.uniform(0.3, 1.0, num_points)
    
    # Create the point array (N, 3)
    points = np.column_stack((x, y, confidence))
    return points

def detect_outliers_by_velocity(x_seg, y_seg, ind_seg, window_size=5, sigma_threshold=2.0):
    """
    Detects outliers based on frame-aware velocity.
    
    Parameters:
    - x_seg, y_seg: x and y coordinates of the segment
    - ind_seg: Frame indices (not uniform)
    - window_size: Number of neighboring points for local statistics
    - sigma_threshold: Outlier threshold based on std dev

    Returns:
    - outlier_indices: Indices of detected outliers within this segment
    """
    num_points = len(x_seg)
    if num_points < 3:
        return []

    # Compute time-aware velocities
    delta_t = np.diff(ind_seg)  # Frame index differences
    delta_x = np.diff(x_seg)
    delta_y = np.diff(y_seg)
    velocity = np.sqrt(delta_x**2 + delta_y**2) / delta_t  # Velocity (Δdistance / Δtime)

    # Compute rolling mean and std deviation of velocity
    local_means = np.array([np.mean(velocity[max(0, i - window_size): i + 1]) for i in range(len(velocity))])
    local_stds = np.array([np.std(velocity[max(0, i - window_size): i + 1]) for i in range(len(velocity))])

    # Outlier detection: velocity > (mean + threshold * std)
    outlier_mask = velocity > (local_means + sigma_threshold * local_stds)

    # Shift because np.diff affects next point
    outlier_indices = np.where(outlier_mask)[0] + 1  

    return outlier_indices

def fit_trajectory_with_confidence(points, confidence_threshold=0.5, smoothing=10, max_gap=30, alpha=0.1):
    """
    Fit a 2D trajectory using B-spline smoothing with confidence-based filtering and adaptive smoothing.

    Parameters:
    - points: (N, 3) NumPy array containing (x, y, confidence)
    - confidence_threshold: Points with confidence below this value are ignored (default: 0.5)
    - base_smoothing: Base smoothing factor (default: 5)
    - max_gap: Maximum allowed gap between frames before splitting (default: 30 frames)
    - alpha: Scaling factor for smoothing adjustment (default: 0.1)

    Returns:
    - all_x_smooth, all_y_smooth: List of smoothed trajectory segments
    - processed_data: (N, 3) array of retained points
    """

    num_points = len(points)
    indices = np.arange(num_points)  # Frame indices as time

    # Step 1: Filter out low-confidence points
    valid_mask = points[:, 2] >= confidence_threshold
    valid_data = points[valid_mask]
    valid_indices = indices[valid_mask]

    if len(valid_data) < 2:
        raise ValueError("Not enough valid points for B-spline fitting!")

    x, y = valid_data[:, 0], valid_data[:, 1]

    processed_data = points.copy()

    # Step 2: Find large gaps and split into segments
    gaps = np.diff(valid_indices)  # Compute gaps between consecutive valid frames
    split_indices = np.where(gaps > max_gap)[0] + 1  # Find large gaps

    ind_segments = np.split(valid_indices, split_indices)
    x_segments = np.split(x, split_indices)
    y_segments = np.split(y, split_indices)

    fitted_indices = []
    # Step 3: Fit B-spline to each segment
    for _, (x_seg, y_seg, ind_seg) in enumerate(zip(x_segments, y_segments, ind_segments)):
        start, end = (ind_seg[0], ind_seg[-1])
        ids = np.arange(start, end+1)
        outliers = detect_outliers_by_velocity(x_seg, y_seg, ind_seg)
        nonoutlier = np.setdiff1d(np.arange(len(x_seg)), outliers)

        outliers = np.setdiff1d(ids, ind_seg[nonoutlier])  # Keep only valid points

        if len(ind_seg) < 2:
            continue  # Skip tiny segments
        # Compute average distance between consecutive points
        avg_distance = np.mean(np.sqrt(np.diff(x_seg)**2 + np.diff(y_seg)**2))
        
        # Adjust smoothing dynamically
        adaptive_s = smoothing * (1 + avg_distance)  # Higher avg_distance → lower s, and vice versa
        
        tck, _ = splprep([x_seg[nonoutlier], y_seg[nonoutlier]], u=ind_seg[nonoutlier], s=adaptive_s)
        print(f"adaptive_s is {adaptive_s}")

        # Generate smooth trajectory for this segment
        x_smooth, y_smooth = splev(outliers, tck)

        valid_fit = (x_smooth > 0) & (y_smooth > 0)
        processed_data[outliers[valid_fit], 0] = x_smooth[valid_fit]
        processed_data[outliers[valid_fit], 1] = y_smooth[valid_fit]
        processed_data[outliers[valid_fit], 2] = 0.51
        fitted_indices.append(outliers[valid_fit])

    return processed_data, fitted_indices

# ============ Test Code ============
if __name__ == "__main__":
    # Generate 30 mock points
    data = generate_mock_data(100)
    data[30:62, -1] = 0.2

    # Fit the trajectory
    processed_data, fitted_indices = fit_trajectory_with_confidence(data, 0.7, smoothing=5)

    # Plot results
    plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1], c=data[:, 2], cmap='coolwarm', edgecolors='k', marker='x', label="Original Points")
    for idx, indices in enumerate(fitted_indices):
        plt.scatter(processed_data[indices, 0], processed_data[indices, 1], c=processed_data[indices, 2], cmap='coolwarm', edgecolors='k', marker='o', label="Processed Points (Retained + Interpolated)")
        
        # Draw lines connecting original to processed points
        for i in indices:
            plt.plot([data[i, 0], processed_data[i, 0]],  # X-coordinates
                    [data[i, 1], processed_data[i, 1]],  # Y-coordinates
                    'gray', linestyle='--', alpha=0.6)   # Dashed gray lines for visibility
    plt.plot(processed_data[:, 0], processed_data[:, 1], 'r--', label=f"B-Spline Fit")
    plt.gca().invert_yaxis()  # Adjust for image coordinate system (origin at top-left)
    plt.colorbar(label="Confidence")
    plt.legend()
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.title("Trajectory Fitting with Index-Based Interpolation")
    plt.show()
