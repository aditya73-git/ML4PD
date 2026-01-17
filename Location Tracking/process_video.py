# import numpy as np
# import cv2
# import pandas as pd
# from ultralytics import YOLO
# # Importiamo le matrici e la funzione dal tuo file originale
# from tracking_code_trial2 import P1, P2, triangulate_point

# # --- CONFIGURAZIONE ---
# VIDEO_CAM1 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_0362.MOV"
# VIDEO_CAM2 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_6590.MOV"
# MODEL_PATH = "runs/detect/train/weights/best.pt"

# # Incolla qui gli intervalli che hai annotato
# # Formato: (frame_inizio, frame_fine)
# throw_intervals = [
#     (6964, 7750), # Lancio 1
#     # (8500, 8640), # Lancio 2
#     # ... aggiungi gli altri qui
# ]

# def run_video_processing():
#     model = YOLO(MODEL_PATH)
#     cap1 = cv2.VideoCapture(VIDEO_CAM1)
#     cap2 = cv2.VideoCapture(VIDEO_CAM2)
#     fps = cap1.get(cv2.CAP_PROP_FPS)
    
#     results_list = []

#     for throw_id, (start, end) in enumerate(throw_intervals, 1):
#         print(f"Processing Throw {throw_id}: Frames {start} to {end}")
        
#         cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
#         cap2.set(cv2.CAP_PROP_POS_FRAMES, start)

#         for current_f in range(start, end + 1):
#             ret1, frame1 = cap1.read()
#             ret2, frame2 = cap2.read()
#             if not ret1 or not ret2: break

#             # Detection su frame (senza salvare su disco per velocità)
#             res1 = model.predict(frame1, conf=0.5, verbose=False)[0]
#             res2 = model.predict(frame2, conf=0.5, verbose=False)[0]

#             if len(res1.boxes) > 0 and len(res2.boxes) > 0:
#                 # Estraiamo i centri dei box (xywh)
#                 c1 = res1.boxes.xywh[0].cpu().numpy()[:2]
#                 c2 = res2.boxes.xywh[0].cpu().numpy()[:2]

#                 # Triangolazione usando la tua funzione importata
#                 X_3d = triangulate_point(P1, P2, c1, c2)

#                 results_list.append({
#                     'throw_id': throw_id,
#                     't': (current_f - start) / fps,
#                     'x_raw': X_3d[0], 'y_raw': X_3d[1], 'z_raw': X_3d[2]
#                 })

#     cap1.release()
#     cap2.release()
    
#     # Creazione DataFrame
#     df = pd.DataFrame(results_list)
    
#     # --- TASK 5a: Generazione Ground Truth tramite Fit Polinomiale ---
#     # Per ogni lancio, fittiamo una parabola su Z e rette su X,Y
#     processed_dfs = []
#     for tid, group in df.groupby('throw_id'):
#         t = group['t'].values
#         # Fit
#         cx = np.polyfit(t, group['x_raw'], 1)
#         cy = np.polyfit(t, group['y_raw'], 1)
#         cz = np.polyfit(t, group['z_raw'], 2)
        
#         # Ground Truth Position (Smooth)
#         group['x_gt'] = np.polyval(cx, t)
#         group['y_gt'] = np.polyval(cy, t)
#         group['z_gt'] = np.polyval(cz, t)
        
#         # Ground Truth Velocity (Analitica)
#         group['vel_x_ground_truth'] = cx[0]
#         group['vel_y_ground_truth'] = cy[0]
#         group['vel_z_ground_truth'] = 2 * cz[0] * t + cz[1]
        
#         processed_dfs.append(group)
    
#     final_df = pd.concat(processed_dfs)
    
#     # --- TASK 5b: Calcolo Velocità Measured (Differenze Finite) ---
#     final_df['vel_x_measured'] = final_df.groupby('throw_id')['x_raw'].diff() / (1/fps)
#     final_df['vel_y_measured'] = final_df.groupby('throw_id')['y_raw'].diff() / (1/fps)
#     final_df['vel_z_measured'] = final_df.groupby('throw_id')['z_raw'].diff() / (1/fps)
    
#     # Esportazione
#     final_df.to_csv("beer_pong_final_dataset.csv", index=False)
#     print("Processing complete. File saved as 'beer_pong_final_dataset.csv'")

# if __name__ == "__main__":
#     run_video_processing()

import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
# Importiamo le matrici e la funzione dal tuo file originale senza modificarlo
from tracking_code_trial2 import P1, P2, triangulate_point

# --- CONFIGURAZIONE ---
VIDEO_CAM1 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_0362.MOV"
VIDEO_CAM2 = "Location Tracking/syncronized_videos/CV_SYNC_IMG_6590.MOV"
MODEL_PATH = "runs/detect/train/weights/best.pt"
OUTPUT_CSV = "beer_pong_trajectories.csv"

# Inserisci qui i tuoi intervalli annotati (start_frame, end_frame)
throw_intervals = [
    (6964, 6990), 
    # (8500, 8640), 
    # ... aggiungi gli altri qui
]

def run_trajectory_extraction():
    model = YOLO(MODEL_PATH)
    cap1 = cv2.VideoCapture(VIDEO_CAM1)
    cap2 = cv2.VideoCapture(VIDEO_CAM2)
    fps = cap1.get(cv2.CAP_PROP_FPS)
    
    raw_data = []

    print("--- Fase 1: Position Tracking & Triangulation ---")
    for throw_id, (start, end) in enumerate(throw_intervals, 1):
        cap1.set(cv2.CAP_PROP_POS_FRAMES, start)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, start)

        for current_f in range(start, end + 1):
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2: break

            res1 = model.predict(frame1, conf=0.4, verbose=False)[0]
            res2 = model.predict(frame2, conf=0.4, verbose=False)[0]

            t_rel = (current_f - start) / fps
            
            # Se la palla è rilevata in entrambe le camere
            if len(res1.boxes) > 0 and len(res2.boxes) > 0:
                c1 = res1.boxes.xywh[0].cpu().numpy()[:2]
                c2 = res2.boxes.xywh[0].cpu().numpy()[:2]
                X_3d = triangulate_point(P1, P2, c1, c2)
                
                raw_data.append({
                    'throw_id': throw_id, 't': t_rel,
                    'x_raw': X_3d[0], 'y_raw': X_3d[1], 'z_raw': X_3d[2]
                })
            else:
                # Se YOLO non vede la palla, mettiamo NaN per interpola dopo
                raw_data.append({
                    'throw_id': throw_id, 't': t_rel,
                    'x_raw': np.nan, 'y_raw': np.nan, 'z_raw': np.nan
                })

    cap1.release()
    cap2.release()
    
    df = pd.DataFrame(raw_data)

    print("--- Fase 2: Interpolazione e Calcolo Coefficienti ---")
    
    # 1. Interpolazione lineare per i buchi di detection (NaN)
    # Fondamentale per non interrompere la traiettoria se YOLO manca un frame
    df[['x_raw', 'y_raw', 'z_raw']] = df.groupby('throw_id')[['x_raw', 'y_raw', 'z_raw']].transform(
        lambda x: x.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    )

    # 2. Regressione per ogni lancio
    processed_list = []
    for tid, group in df.groupby('throw_id'):
        t = group['t'].values
        
        # Fit X, Y (lineare) e Z (parabolico)
        cx = np.polyfit(t, group['x_raw'], 1)
        cy = np.polyfit(t, group['y_raw'], 1)
        cz = np.polyfit(t, group['z_raw'], 2)
        
        # Salviamo i coefficienti come colonne costanti per quel throw_id
        group['coeff_x_1'], group['coeff_x_0'] = cx[0], cx[1]
        group['coeff_y_1'], group['coeff_y_0'] = cy[0], cy[1]
        group['coeff_z_2'], group['coeff_z_1'], group['coeff_z_0'] = cz[0], cz[1], cz[2]
        
        processed_list.append(group)

    final_df = pd.concat(processed_list)
    
    # Output pulito: id, tempo, posizioni raw interpolate, e coefficienti
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset creato con successo: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_trajectory_extraction()

    # Print trajectories over x, y and z, showing both raw and fitted data
    import matplotlib.pyplot as plt
    df = pd.read_csv(OUTPUT_CSV)
    for throw_id, group in df.groupby('throw_id'):
        t = group['t'].values
        x_fit = group['coeff_x_1'].iloc[0] * t + group['coeff_x_0'].iloc[0]
        y_fit = group['coeff_y_1'].iloc[0] * t + group['coeff_y_0'].iloc[0]
        z_fit = group['coeff_z_2'].iloc[0] * t**2 + group['coeff_z_1'].iloc[0] * t + group['coeff_z_0'].iloc[0]

        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(t, group['x_raw'], 'o', label='Raw X')
        plt.plot(t, x_fit, '-', label='Fitted X')
        plt.xlabel('Time (s)')
        plt.ylabel('X Position (m)')
        plt.title(f'Throw {throw_id} - X Position')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(t, group['y_raw'], 'o', label='Raw Y')
        plt.plot(t, y_fit, '-', label='Fitted Y')
        plt.xlabel('Time (s)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Throw {throw_id} - Y Position')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(t, group['z_raw'], 'o', label='Raw Z')
        plt.plot(t, z_fit, '-', label='Fitted Z')
        plt.xlabel('Time (s)')
        plt.ylabel('Z Position (m)')
        plt.title(f'Throw {throw_id} - Z Position')
        plt.legend()

        plt.tight_layout()
        plt.show()