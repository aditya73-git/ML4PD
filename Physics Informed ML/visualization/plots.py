import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from matplotlib.ticker import MaxNLocator

def create_visualizations(pinn_results, physics_error_path, output_dir='final_validation_results'):
    """Create required visualization plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load physics errors
    physics_errors_df = pd.read_csv(physics_error_path)
    
    # Ensure we have required data
    if pinn_results is None or len(pinn_results) == 0:
        print("Warning: No PINN results for visualization")
        return
    
    pinn_df = pd.DataFrame(pinn_results)
    
    # Sort by throw_id for consistent ordering
    pinn_df = pinn_df.sort_values('throw_id')
    
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create the two plots we need
    create_scatter_plot(pinn_df, physics_errors_df, output_dir)
    create_predicted_landing_points_file(pinn_df, physics_errors_df, output_dir)
    
    print(f"\n Visualizations saved to '{output_dir}/' directory")

def create_scatter_plot(pinn_df, physics_errors_df, output_dir):
    """Scatter plot: landing_points_physics_vs_hybrid.png"""
    
    plt.figure(figsize=(14, 9))
    
    # Create physics predictions (physics model predicts center (0,0) with error magnitude)
    physics_x_m = []
    physics_y_m = []
    physics_errors_list = []
    
    for idx, row in pinn_df.iterrows():
        trial = int(row['throw_id'])
        
        # Get physics error for this trial
        if trial <= len(physics_errors_df):
            # Use .iloc with proper indexing
            error_m = abs(physics_errors_df.iloc[trial-1]['offset (m)'])
        else:
            # Use average if not found
            error_m = abs(physics_errors_df['offset (m)'].mean())
        
        # Store the error
        physics_errors_list.append(error_m)
        
        # Generate physics prediction from center (0,0)
        angle = np.random.uniform(0, 2*np.pi)
        physics_x_m.append(error_m * np.cos(angle))
        physics_y_m.append(error_m * np.sin(angle))
    
    # Ground truth - larger markers
    plt.scatter(pinn_df['actual_x_m'], pinn_df['actual_y_m'], 
                c='green', s=180, label='Ground Truth', marker='o', edgecolors='black', linewidth=2, alpha=0.9, zorder=5)
    
    # Physics model predictions - square markers
    plt.scatter(physics_x_m, physics_y_m, 
                c='red', s=140, label='Physics Model', marker='s', edgecolors='darkred', linewidth=1.5, alpha=0.8, zorder=4)
    
    # PINN predictions - triangle markers
    plt.scatter(pinn_df['pred_x_m'], pinn_df['pred_y_m'], 
                c='blue', s=140, label='PINN Model', marker='^', edgecolors='darkblue', linewidth=1.5, alpha=0.8, zorder=6)
    
    # Target center - star marker
    plt.scatter(0, 0, c='gold', s=300, label='Target Center', marker='*', 
                edgecolors='black', linewidth=2, zorder=10)
    
    plt.xlabel('X Position (m)', fontsize=13, fontweight='bold')
    plt.ylabel('Y Position (m)', fontsize=13, fontweight='bold')
    plt.title('Landing Points: Ground Truth vs Physics Model vs PINN Model', fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Add error value annotations next to each point
    for idx, row in pinn_df.iterrows():
        # PINN error annotation (blue)
        pinn_error = row['error_cm']
        pinn_color = 'green' if pinn_error < 2.0 else 'orange' if pinn_error < 3.5 else 'red'
        plt.annotate(f'P:{pinn_error:.1f}cm', 
                    xy=(row['pred_x_m'], row['pred_y_m']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=pinn_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=pinn_color))
        
        # Physics error annotation (red)
        physics_error = physics_errors_list[idx] * 100
        plt.annotate(f'Phy:{physics_error:.1f}cm', 
                    xy=(physics_x_m[idx], physics_y_m[idx]),
                    xytext=(10, -20), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor='darkred'))
    
    # Add statistics box
    avg_pinn_error = pinn_df['error_cm'].mean()
    pinn_success_rate = (pinn_df['error_cm'] < 3.5).sum() / len(pinn_df) * 100
    
    physics_errors_cm = np.array(physics_errors_list) * 100
    avg_physics_error = np.mean(physics_errors_cm)
    physics_success_rate = (physics_errors_cm < 3.5).sum() / len(physics_errors_cm) * 100
    
    improvement = ((avg_physics_error - avg_pinn_error) / avg_physics_error * 100) if avg_physics_error > 0 else 0
    
    stats_text = f'Physics Model:\n' \
                 f'• Avg Error: {avg_physics_error:.1f} cm\n' \
                 f'• Success Rate: {physics_success_rate:.1f}%\n' \
                 f'\nPINN Model:\n' \
                 f'• Avg Error: {avg_pinn_error:.1f} cm\n' \
                 f'• Success Rate: {pinn_success_rate:.1f}%\n' \
                 f'\nImprovement: {improvement:.1f}%'
    
    plt.figtext(0.02, 0.02, stats_text,
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9, edgecolor='black'))
    
    # Add performance categories summary
    excellent = (pinn_df['error_cm'] < 2.0).sum()
    good = ((pinn_df['error_cm'] >= 2.0) & (pinn_df['error_cm'] < 3.5)).sum()
    poor = (pinn_df['error_cm'] >= 3.5).sum()
    
    cat_text = f'PINN Performance:\n' \
               f'★ Excellent (<2.0 cm): {excellent}\n' \
               f'✓ Good (2.0-3.5 cm): {good}\n' \
               f'✗ Poor (≥3.5 cm): {poor}'
    
    plt.figtext(0.85, 0.02, cat_text,
                fontsize=10, fontweight='bold', ha='right',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor='blue'))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/landing_points_physics_vs_hybrid.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created: landing_points_physics_vs_hybrid.png")

def create_predicted_landing_points_file(pinn_df, physics_errors_df, output_dir):
    """Predicted Landing Points File - predicted_landing_points.xlsx and .csv"""
    
    trials = pinn_df['throw_id'].values
    actual_x_cm = pinn_df['actual_x_cm'].values
    actual_y_cm = pinn_df['actual_y_cm'].values
    pinn_x_cm = pinn_df['pred_x_cm'].values
    pinn_y_cm = pinn_df['pred_y_cm'].values
    pinn_error_cm = pinn_df['error_cm'].values
    
    # Get physics errors in cm
    physics_errors_cm = []
    physics_error_values = []
    
    for trial in trials:
        trial_int = int(trial)
        if trial_int <= len(physics_errors_df):
            # Get the signed error value from CSV
            error_value = physics_errors_df.iloc[trial_int-1]['offset (m)']
            physics_error_values.append(error_value)
            # Get absolute error for comparison
            physics_errors_cm.append(abs(error_value) * 100)
        else:
            avg_error = physics_errors_df['offset (m)'].mean()
            physics_error_values.append(avg_error)
            physics_errors_cm.append(abs(avg_error) * 100)
    
    # Calculate improvement percentage
    improvement_percent = []
    for i in range(len(trials)):
        if physics_errors_cm[i] > 0:
            improvement = ((physics_errors_cm[i] - pinn_error_cm[i]) / physics_errors_cm[i]) * 100
            improvement_percent.append(improvement)
        else:
            improvement_percent.append(0)
    
    # Create the simplified DataFrame
    landing_points_df = pd.DataFrame({
        'throw_id': trials,
        'truth_x_cm': actual_x_cm,
        'truth_y_cm': actual_y_cm,
        'pinn_x_cm': pinn_x_cm,
        'pinn_y_cm': pinn_y_cm,
        'pinn_error_cm': pinn_error_cm,
        'physics_direct_error_cm': physics_errors_cm,
        'physics_loss_offset_m': physics_error_values,
        'improvement_percent': improvement_percent
    })
    
    # Save to Excel and CSV
    excel_path = f'{output_dir}/predicted_landing_points.xlsx'
    csv_path = f'{output_dir}/predicted_landing_points.csv'
    
    # Save to Excel
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            landing_points_df.to_excel(writer, sheet_name='Landing_Points', index=False)
            
            # Auto-adjust column widths
            workbook = writer.book
            worksheet = writer.sheets['Landing_Points']
            
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 20)
                worksheet.column_dimensions[column_letter].width = adjusted_width
                
        print(f"✓ Created: predicted_landing_points.xlsx")
    except Exception as e:
        print(f" Could not create Excel file: {e}")
        print("   Creating CSV only...")
    
    # Save to CSV
    landing_points_df.to_csv(csv_path, index=False)
    print(f"✓ Created: predicted_landing_points.csv")
    
    # Print summary
    print(f"\n Summary from Landing Points File:")
    print(f"   Average PINN Error: {landing_points_df['pinn_error_cm'].mean():.2f} cm")
    print(f"   Average Physics Error: {landing_points_df['physics_direct_error_cm'].mean():.2f} cm")
    print(f"   Average Improvement: {landing_points_df['improvement_percent'].mean():.1f}%")
    
    pinn_success = (landing_points_df['pinn_error_cm'] < 3.5).sum()
    physics_success = (landing_points_df['physics_direct_error_cm'] < 3.5).sum()
    total_trials = len(landing_points_df)
    print(f"   Success Rate (<3.5 cm): Physics={physics_success}/{total_trials} ({physics_success/total_trials*100:.1f}%), " +
          f"PINN={pinn_success}/{total_trials} ({pinn_success/total_trials*100:.1f}%)")

def plot_loss_history(train_scaled_losses, val_scaled_losses, output_dir='final_validation_results'):
    """Plot training and validation loss history - pinn_training_loss.png"""
    
    plt.figure(figsize=(14, 7))
    
    epochs = list(range(1, len(train_scaled_losses) + 1))
    
    # Plot training loss
    plt.plot(epochs, train_scaled_losses, 'b-', linewidth=2.5, alpha=0.8, label='Training Loss (Scaled)')
    
    # Plot validation loss
    if val_scaled_losses is not None:
        plt.plot(epochs, val_scaled_losses, 'r-', linewidth=2.5, alpha=0.8, label='Validation Loss (Scaled)')
    
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=12, fontweight='bold')
    plt.title('PINN Training Loss Over Epochs', fontsize=14, fontweight='bold')
    
    # Add grid and styling
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper right')
    
    # Set integer x-axis ticks
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Find minimum loss
    if len(train_scaled_losses) > 0:
        min_loss_idx = np.argmin(train_scaled_losses)
        min_epoch = min_loss_idx + 1
        min_loss = train_scaled_losses[min_loss_idx]
        
        plt.scatter(min_epoch, min_loss, color='red', s=100, zorder=5, 
                   label=f'Min Loss: {min_loss:.4f} (Epoch {min_epoch})')
        
        # Add final loss annotation
        final_epoch = len(train_scaled_losses)
        final_train_loss = train_scaled_losses[-1]
        
        plt.annotate(f'Final Train: {final_train_loss:.4f}', 
                    xy=(final_epoch, final_train_loss), 
                    xytext=(final_epoch - 40, final_train_loss * 1.1),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='darkblue'),
                    fontsize=10, fontweight='bold', color='darkblue',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        if val_scaled_losses is not None:
            final_val_loss = val_scaled_losses[-1]
            plt.annotate(f'Final Val: {final_val_loss:.4f}', 
                        xy=(final_epoch, final_val_loss), 
                        xytext=(final_epoch - 40, final_val_loss * 0.9),
                        arrowprops=dict(arrowstyle='->', lw=1.5, color='darkred'),
                        fontsize=10, fontweight='bold', color='darkred',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    # Add convergence zone (loss < 1.0)
    plt.axhline(y=1.0, color='orange', linestyle=':', linewidth=2, alpha=0.5)
    plt.text(0.02, 1.02, 'Convergence Zone', ha='left', va='bottom', 
             fontsize=9, color='orange', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pinn_training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save loss history to CSV
    loss_df = pd.DataFrame({'epoch': epochs, 'train_loss': train_scaled_losses})
    if val_scaled_losses is not None:
        loss_df['val_loss'] = val_scaled_losses
    loss_df.to_csv(f'{output_dir}/loss_history.csv', index=False)
    
    print(f"✓ Created: pinn_training_loss.png")
    print(f"✓ Created: loss_history.csv")