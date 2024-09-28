import cv2

def split_video_horizontally(input_video_path, output_video_prefix):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the height of each horizontal piece
    piece_height = frame_height // 4

    # Define the codec and create VideoWriter objects for each piece
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(f'{output_video_prefix}_part1.mp4', fourcc, fps, (frame_width, piece_height))
    out2 = cv2.VideoWriter(f'{output_video_prefix}_part2.mp4', fourcc, fps, (frame_width, piece_height))
    out3 = cv2.VideoWriter(f'{output_video_prefix}_part3.mp4', fourcc, fps, (frame_width, piece_height))
    out4 = cv2.VideoWriter(f'{output_video_prefix}_part4.mp4', fourcc, fps, (frame_width, piece_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Split the frame into four horizontal pieces
        piece1 = frame[:piece_height, :]
        piece2 = frame[piece_height:2*piece_height, :]
        piece3 = frame[2*piece_height:3*piece_height, :]
        piece4 = frame[3*piece_height:, :]

        # Write the pieces to the output video files
        out1.write(piece1)
        out2.write(piece2)
        out3.write(piece3)
        out4.write(piece4)

    # Release everything if job is finished
    cap.release()
    out1.release()
    out2.release()
    out3.release()
    out4.release()

# Example usage
input_video_path = '/home/aleximu/gunes/dinov2/project/signal-2024-05-24-162625_003.mp4'
output_video_prefix = 'output_video'
split_video_horizontally(input_video_path, output_video_prefix)