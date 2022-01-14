import bitarray


def haswon(board):
	print(board)
	print(board >> 7)
	y = board & (board >> 7)
	print(y)
	print((y >> 2 * 7))
	if (y & (y >> 2 * 7)):
		return 1
	y = board & (board >> 8)
	if (y & (y >> 2 * 8)):
		return 1
	y = board & (board >> 9)
	if (y & (y >> 2 * 9)):
		return 1
	y = board & (board >> 1)
	if (y & (y >> 2)):
		return 1
	return 0

a = bitarray.bitarray('11100000110000000101000010100000110000000001000001010000')

print(haswon(a))
