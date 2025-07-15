from moviepy.editor import VideoFileClip

VIDEO_PATH = "C:\\Users\\pzt0029\\Documents\\Classes\\COMP_6650_Deep_Learning\\Project\\videos\\LRMC.mp4"
GIF_PATH = "C:\\Users\\pzt0029\\Documents\\Classes\\COMP_6650_Deep_Learning\\Project\\Images\\ff_lrmc.gif"

video = VideoFileClip(VIDEO_PATH)
video.write_gif(GIF_PATH)