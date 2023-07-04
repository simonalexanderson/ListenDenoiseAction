# import pafy

# #URL of the YouTube video
# url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# #Create a pafy object
# video = pafy.new(url)

# #Get the best audio quality available
# best_audio = video.getbestaudio()

# #Download the audio file
# best_audio.download()

# #Convert the downloaded audio file to wav format using ffmpeg
# import subprocess
# subprocess.call(["ffmpeg", "-i", best_audio.title, "-acodec", "pcm_s16le", "-ar", "44100", "output.wav"])

import youtube_dl

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
    'key': 'FFmpegExtractAudio',
    'preferredcodec': 'wav',
    'preferredquality': '192',
    }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=EXJx2NnnxA0'])
    ydl.download(['https://www.youtube.com/watch?v=-tWUlOt7V8I'])
    ydl.download(['https://www.youtube.com/watch?v=oF69sSGzje8'])    
    ydl.download(['https://www.youtube.com/watch?v=QE5D2hJhacU'])
    ydl.download(['https://www.youtube.com/watch?v=god7hAPv8f0'])
    ydl.download(['https://www.youtube.com/watch?v=KBn_oUH8Uo0'])    
    ydl.download(['https://www.youtube.com/watch?v=MedNNDFaAFU'])
    ydl.download(['https://www.youtube.com/watch?v=jyvH6wf4ghw'])
    ydl.download(['https://www.youtube.com/watch?v=cJTZnHkldW8'])
    ydl.download(['https://www.youtube.com/watch?v=CNJomxuOncg'])
    ydl.download(['https://www.youtube.com/watch?v=fL_daMybahQ'])
    ydl.download(['https://www.youtube.com/watch?v=O2G32MyUwcc'])
    ydl.download(['https://www.youtube.com/watch?v=yai7wWLl104'])
    ydl.download(['https://www.youtube.com/watch?v=2W6uPpQnxn4'])
    ydl.download(['https://www.youtube.com/watch?v=Aeynzbdbsk4'])
    ydl.download(['https://www.youtube.com/watch?v=jZWWYBEtZSo'])
    ydl.download(['https://www.youtube.com/watch?v=cb2w2m1JmCY'])
    ydl.download(['https://www.youtube.com/watch?v=_WXTukojxus&list=PLIvacmZCzEbDEjwerGBLVYiMowyuELPTa&index=15'])
    ydl.download(['https://www.youtube.com/watch?v=f80okiRz_qo&list=PLIvacmZCzEbB4ydxYDW_zhblZSYrHA2U4'])
    ydl.download(['https://www.youtube.com/watch?v=V_j0t6CCto4&list=PLIvacmZCzEbB4ydxYDW_zhblZSYrHA2U4&index=5'])
    ydl.download(['https://www.youtube.com/watch?v=7lPqkPpuEQ4'])
    ydl.download(['https://www.youtube.com/watch?v=URT9aYTRSRo'])
    ydl.download(['https://www.youtube.com/watch?v=URT9aYTRSRo'])
    ydl.download(['https://www.youtube.com/watch?v=NxNb8zYXwL4'])
    ydl.download(['https://www.youtube.com/watch?v=BE_IodYmCC0'])
    ydl.download(['https://www.youtube.com/watch?v=2C7NSOop-HM'])
