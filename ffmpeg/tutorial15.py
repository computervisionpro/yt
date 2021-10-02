import ffmpeg
import sys

sys.path.append(r'C:\Program Files\ffmpeg-4.3\bin') # your ffmpeg file path

stream = ffmpeg.input('sea-video.mp4') # video location

stream = stream.trim(start = 10, duration=15).filter('setpts', 'PTS-STARTPTS')
stream = stream.filter('fps', fps=5, round='up').filter('scale', w=128, h=128)

stream = ffmpeg.output(stream, 'output.mp4')

ffmpeg.run(stream)

