#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

//Reference implementation of Jacobi for mp2 (sequential)
//Constants are being used instead of arguments
#define BC_HOT  1.0
#define BC_COLD 0.0
#define INITIAL_GRID 0.5
#define TOL 1.0e-8
#define ARGS 5
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

struct timeval tv;
double get_clock() {
	struct timeval tv; int ok;
	ok = gettimeofday(&tv, (void *) 0);
	if (ok<0) { printf("gettimeofday error");  }
	return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}


double **
create_matrix(int h, int w) {
	int i;
	double **a;

	a = (double**) malloc(sizeof(double*)*h);
	for (i=0;i<h;i++) {
		a[i] = (double*) malloc(w * sizeof(double));
	}

	return a;
}

void
mark (int rank)
{
	static int i = 0;
	/*
	   printf("[%d] %d\n", rank, i++);
	   */
	if (rank == 0)
		printf("%d\n", i++);
}

void
init_matrix(double **a, int h, int w, int hotlen, int coldlen, int rank, int size)
{
	int i, j;
	for(i=0; i<h; i++) {
		for(j=0; j<w; j++)
			a[i][j] = INITIAL_GRID;
	}
	for(i=0;i<hotlen;i++)           /* Hot boundary */
		a[i][0] = BC_HOT;
	if (rank == size-1)
		for(j=coldlen;j<w;j++)    /* Cold boundary */
			a[h-1][j] = BC_COLD;
}

void
free_matrix (double **a, int h)
{
	int i;
	for (i=0;i<h;i++) {
		free(a[i]);
	}
	free(a);
}

void
matcpy (double **d, double **s, int h, int w)
{
	int i;
	for (i = 0; i < h; i++)
		memcpy(d[i], s[i], w * sizeof (double));
}

void
swap_matrix(double ***a, double ***b)
{
	double **temp = *a;
	*a = *b;
	*b = temp;
}

void
print_grid(double **a, int h, int w)
{
	int i, j;
	for(i=0; i<h; i++) {
		for(j=0; j<w; j++) {
			printf("%6.4lf ",a[i][j]);
		}
		printf("\n");
	}
}

int
boundary_init (double **a, int hotlen, int coldlen, int n, int h, int w, int rank, int size)
{
	MPI_Request req[4] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL, MPI_REQUEST_NULL};
	int i, j;
	/* TOP */
	if (!rank)
		for (j = 0; j < w+2; j++)
			a[0][j] = a[1][j];
	/* BOTTOM */
	if (rank == size-1)
		for (j = 0; j < coldlen; j++)
			a[h+1][j] = a[h][j];
	/* LEFT */
	for (i = hotlen; i < h+2; i++)
		a[i][0] = a[i][1];
	/* RIGHT */
	for (i = 0; i < h+2; i++)
		a[i][n+1] = a[i][n];

	if (rank != size-1) {
		MPI_Irecv(&a[h+1][1], w, MPI_DOUBLE, rank+1, MPI_ANY_TAG,
				MPI_COMM_WORLD, &req[1]);
		MPI_Isend(&a[h][1], w, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[2]);
	}
	if (rank) {
		MPI_Irecv(&a[0][1], w, MPI_DOUBLE, rank-1, MPI_ANY_TAG,
				MPI_COMM_WORLD, &req[0]);
		MPI_Isend(&a[1][1], w, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[3]);
	}
	MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

	return 0;
}

double
compute_grid (double **a, double **b, int h, int w, int rank)
{
	int i, j;
	int imax = 0, jmax = 0;
	double maxdiff_loc = 0.0;
	for (i = 1; i < h+1; i++)
		for (j = 1; j < w+1; j++) {
			b[i][j] = 0.2*(a[i][j] + a[i-1][j] + a[i+1][j] +
					a[i][j-1] + a[i][j+1]);
			if (fabs(b[i][j] - a[i][j]) > maxdiff_loc) {
				maxdiff_loc = fabs(b[i][j] - a[i][j]);
				imax = i; jmax = j;
			}
		}
	/*printf("[%d] \tTolerance = %12.10lf \tat (%d, %d) \t=> %12.10lf to %12.10lf\n",
		iPI_Irecv(&a[h+1][1], w, MPI_DOUBLE, rank-1, MPI_ANY_TAG,
				MPI_COMM_WORLD, &req[1]);
			rank, maxdiff_loc, imax+rank*h, jmax, a[imax][jmax], b[imax][jmax]);
			*/
	return maxdiff_loc;
}

void
gather_and_printgrid(double **a, int n, int h, int w, int rank, int size)
{
	double **result, **res;
	int p, i, j;

	/*
	for (p = 0; p < size; p++)
	{
		if (p == rank) {
			printf("[GROUP %d]\n", p);
			for (i = 0; i < h+2; i++)
			{
				for (j = 0; j < w+2; j++)
				{
					printf("%6.4lf ",a[i][j]);
				}
				putchar('\n');
			}
			putchar('\n');
		}
			fflush(stdout);
		MPI_Barrier(MPI_COMM_WORLD);
	}
	*/

	if (rank) {
		for (i = 0; i < h+2; i++)
			MPI_Send(&a[i][0], w+2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	} else {
		res = create_matrix(h+2, w+2);
		result = create_matrix(n+2, n+2);
		matcpy(result, a, h+2, w+2);
		printf("[GROUP 0]\n");
		print_grid(a, h+2, w+2);
		for (i = 1; i < size; i++)
		{
			for (j = 0; j < h+2; j++)
			{
				MPI_Recv(res[j], w+2, MPI_DOUBLE, i, MPI_ANY_TAG,
						MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			printf("[GROUP %d]\n", i);
			print_grid(res, h+2, w+2);
		}
		putchar('\n');
		free_matrix(result, n+2);
		free_matrix(res, h+2);
	}
}

int
pjacobi (int n, int r, int c, int max_iterations, int rank, int size)
{
	double tstart, tend, ttotal;	/* Runtime data */
	int iteration = 0;
	double **a, **b;                /* Matrix data */
	double maxdiff = 1.0;
	int h = n/size,
	    w = n,
	    hotlen, coldlen;

	hotlen = MAX(MIN(h, n/2 - h*rank), 0);
	if (hotlen == h)	hotlen++;
	if (hotlen)		hotlen++;
	coldlen = n/2;
	if (coldlen == h)	coldlen++;
	if (coldlen)		coldlen++;

	/* INIT */
	a = create_matrix(h+2, w+2);
	b = create_matrix(h+2, w+2);
	init_matrix(a,h+2, w+2, hotlen, coldlen, rank, size);
	matcpy(b, a, h+2, w+2);

	MPI_Barrier(MPI_COMM_WORLD);
	tstart = get_clock();
	if (!rank) {
		printf("Running simulation with tolerance=%lf and max iterations=%d\n",
				TOL, max_iterations);
	}

	while (TOL < maxdiff && iteration++ < max_iterations)
	{
		double maxdiff_loc;
		boundary_init(a, hotlen, coldlen, n, h, w, rank, size);
		maxdiff_loc = compute_grid(a, b, h, w, rank);
		swap_matrix(&a,&b);
		MPI_Allreduce(&maxdiff_loc, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if (!rank) {
			//printf("[ALL] \tTolerance =%12.10lf\n", maxdiff);
		}
	}

	if (rank == r/h) {
		tend = get_clock();
		ttotal = tend-tstart;
		printf("Results:\n");
		printf("Iterations=%d\n",iteration-1);
		printf("Tolerance=%12.10lf\n",maxdiff);
		printf("Running time=%12.8lf\n",ttotal);
		printf("Value at (%d,%d)=%12.8lf\n",r,c,a[r-rank*h][c]);
	}

	free_matrix(a, h+2);
	free_matrix(b, h+2);
	return 0;
}

int
main (int argc, char *argv[])
{
	int n, r, c, max_iterations;    /* Parameters */
	int rank, size;                 /* MPI variables */
	int status;

	/* Start MPI routine */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc != ARGS) {
		if (!rank)
			fprintf(stderr,"Wrong # of arguments.\nUsage: %s N I R C\n",
					argv[0]);
		status = -1;
	} else {
		n = atoi(argv[1]);
		max_iterations = atoi(argv[2]);
		r = atoi(argv[3]);
		c = atoi(argv[4]);
		status = pjacobi(n, r, c, max_iterations, rank, size);
	}

	MPI_Finalize();

	return status;
}
