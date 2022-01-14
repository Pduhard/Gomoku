
int haswon(long board)
{
    long y = board & (board >> 7);
    if (y & (y >> 2 * 7)) // check \ diagonal
        return 1;
		
    y = board & (board >> 8);
    if (y & (y >> 2 * 8)) // check horizontal -
        return 1;
    y = board & (board >> 9);
    if (y & (y >> 2 * 9)) // check / diagonal
        return 1;
    y = board & (board >> 1);
    if (y & (y >> 2))     // check vertical |
        return 1;
    return 0;
}

int main()
{
	int a = 0x08;

	long board = 0b11100000110000000101000010100000110000000001000001010000;
	printf("%d\n", haswon(board));
	// *0b
	// 11100000
	// 11000000
	// 01010000
	// 11100000
	// 11000000
	// 01010000
	// 01010000

	printf("%d %d %d\n", a<<4*2, (a<<4)*2, a<<(8));
}