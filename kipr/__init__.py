from IPython.display import Audio

def sys_bgcolor(pyplot):
		"""Reads system preference and sets plt background accordingly"""
		from winreg import ConnectRegistry, HKEY_CURRENT_USER, OpenKeyEx, QueryValueEx
		root = ConnectRegistry(None, HKEY_CURRENT_USER)
		policy_key = OpenKeyEx(root, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize")
		light, _ = QueryValueEx(policy_key, "AppsUseLightTheme")
		if light:
			pyplot.style.use('default')
		else:
			pyplot.style.use('dark_background')

def winplay(data, rate, *, width=3, offset=0, seconds=15):
	"""Plays a numpy array as a sound on Windows (and is unstoppable)"""
	import winsound
	data = data[rate*offset:rate*(offset + seconds)]
	r = bytearray()
	r += b'\x52\x49\x46\x46' # RIFF
	r += (width*len(data) + 36).to_bytes(4, 'little') # bytes until EOF
	r += b'\x57\x41\x56\x45' # WAVE
	r += b'\x66\x6D\x74\x20' # fmt_
	r += b'\x10\x00\x00\x00' # 16 bytes for the chunk
	r += b'\x01\x00' # PCM = 1
	r += b'\x01\x00' # number of channels
	r += (rate).to_bytes(4, 'little') # number of channels
	r += (rate*width).to_bytes(4, 'little') # bytes per sec
	r += (width).to_bytes(2, 'little') # bytes per block
	r += (width*8).to_bytes(2, 'little') # bits per sample
	r += b'\x64\x61\x74\x61' # data
	r += (width*len(data)).to_bytes(4, 'little') # bytes until EOF
	max_value = max(abs(data))
	max_number = 2**(width*8 - 1) - 1
	for e in data:
		r += int(e*max_number/max_value).to_bytes(width, 'little', signed=True)
	winsound.PlaySound(r, winsound.SND_MEMORY)

def finished():
	"""To play a sound after execution"""
	return Audio('https://www.soundboard.com/mediafiles/23/230637-88d7c1eb-fd29-4c12-9775-f8dff855374b.mp3',
		         autoplay=True)