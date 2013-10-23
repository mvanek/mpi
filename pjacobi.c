#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <openmpi/mpi.h>

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


double **create_matrix(int h, int w) {
	int i;
	double **a;

	a = (double**) malloc(sizeof(double*)*h);
	for (i=0;i<h;i++) {
		a[i] = (double*) malloc(w * sizeof(double));
	}

	return a;
}

void init_matrix(double **a, int h, int w)
{
	int i, j;
	for(i=0; i<h; i++) {
		for(j=0; j<w; j++)
			a[i][j] = INITIAL_GRID;
	}
}

void free_matrix(double **a, int h)
{
	int i;
	for (i=0;i<h;i++) {
		free(a[i]);
	}
	free(a);
}

void matcpy (double **d, double **s, int h, int w)
{
    int i;
    for (i = 0; i < h; i++)
        memcpy(d[i], s[i], w * sizeof (double));
}

void swap_matrix(double ***a, double ***b)
{
	double **temp = *a;
	*a = *b;
	*b = temp;
}

void print_grid(double **a, int h, int w)
{
	int i, j;
	for(i=1; i<h-1; i++) {
		for(j=1; j<w-1; j++) {
			printf("%6.4lf ",a[i][j]);
		}
		printf("\n");
	}
}

void
MPI_perror(int errno, char *s)
{
    char out[MPI_MAX_ERROR_STRING];
    char *format = "%s: %.*s\n";
    int bytes;
    if (MPI_Error_string(errno, out, &bytes)) {
        fprintf(stderr, "MPI_perror: bad error value\n");
        return;
    }
    if (s)
        format += 4;
    fprintf(stderr, format, s, bytes, out);
}

void
mark (int rank)
{
    static int i = 0;
    /*
    printf("[%d] %d\n", rank, i++);
    */
    if (rank == 2)
        printf("%d\n", i++);
}

int
main (int argc, char *argv[])
{
    int n, r, c, max_iterations;    /* Parameters */
    int bytes, status;              /* Return values */

    int rank, size;                 /* MPI variables */

    double **a, **b;                /* The matrices */
    double maxdiff, gmaxdiff;
    int bclength;
    int i, j, h, w;

	double tstart, tend, ttotal;
    int iteration;

	if (argc != ARGS) {
		fprintf(stderr,"Wrong # of arguments.\nUsage: %s N I R C\n",
					argv[0]);
		return -1;
	}
	n = atoi(argv[1]);
	max_iterations = atoi(argv[2]);
	r = atoi(argv[3]);
	c = atoi(argv[4]);

    /* Start MPI routine */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Assign rows */
    h = n / size;
    w = n;
	bclength = (n+2)/2;

	a = create_matrix(h+2, w+2);
	b = create_matrix(h+2, w+2);
	init_matrix(a,h+2, w+2);

	/* Initialize the hot boundary */
	for(i=0;i<MIN(h, bclength-rank*h);i++)
		a[i][0] = BC_HOT;

	/* Initialize the cold boundary */
	for(j=MAX(0, bclength-rank*h);j<w+2;j++)
		a[h+1][j] = BC_COLD;

	/* Copy a to b */
	for(i=0; i<h+2; i++)
		for(j=0; j<w+2; j++)
			b[i][j] = a[i][j];

	iteration = 0;
	gmaxdiff = maxdiff = 1.0;
	printf("Running simulation with tolerance=%lf and max iterations=%d\n",
		TOL, max_iterations);
	tstart = get_clock();
	while (TOL < gmaxdiff && iteration++ < max_iterations)
    {
		/* Initialize boundary values */
        /* TOP */
        if (!rank) {        /* The top block */
            for (j = 0; j < w+2; j++)
                a[0][j] = a[1][j];
        } else {
            if ((status = MPI_Recv(&a[0][1], w, MPI_DOUBLE,
                            MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE)))
            {
                MPI_perror(status, "MPI_Recv");
                return -1;
            }
        }
        if (rank != size-1) {
            if ((status = MPI_Ssend(&a[h][1], w, MPI_DOUBLE, rank+1,
                            0, MPI_COMM_WORLD)))
            {
                MPI_perror(status, "MPI_Send");
                return -1;
            }
        }

        /* BOTTOM */
        if (rank == size-1)     /* The bottom block */
            for (j = 0; j < w+2-bclength; j++)
                a[h+1][j] = a[h][j];
        else if ((status = MPI_Recv(&a[h+1][1], w, MPI_DOUBLE,
                            MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE)))
            {
                MPI_perror(status, "MPI_Recv");
                return -1;
            }
        if (rank)
            if ((status = MPI_Send(&a[1][1], w, MPI_DOUBLE, rank-1,
                            0, MPI_COMM_WORLD)))
            {
                MPI_perror(status, "MPI_Send");
                return -1;
            }

		/* LEFT */
		for (i = MAX(bclength - rank*h, 0); i < h+2; i++)
			a[i][0] = a[i][1];

		/* RIGHT */
		for (i = 0; i < h+2; i++)
			a[i][n+1] = a[i][n];

		/* Compute new grid values */
		maxdiff = 0.0;
		for (i = 1; i < h+1; i++)
			for (j = 1; j < w+1; j++) {
				b[i][j] = 0.2*(a[i][j] + a[i-1][j] + a[i+1][j] +
					       a[i][j-1] + a[i][j+1]);
				if (fabs(b[i][j] - a[i][j]) > maxdiff)
					maxdiff = fabs(b[i][j] - a[i][j]);
			}

		/* Copy b to a */
		swap_matrix(&a,&b);
        MPI_Allreduce(&maxdiff, &gmaxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}

    if (rank) {
        for (i = 1; i < h+1; i++)
            if ((status = MPI_Send(&a[i][1], w, MPI_DOUBLE, 0,
                            0, MPI_COMM_WORLD)))
            {
                MPI_perror(status, "MPI_Send");
                return -1;
            }
    } else {
        double **result = create_matrix(n, n);
        matcpy(result, &a[1], h, w);
        for (i = 1; i < size; i++)
            for (j = 0; j < h; j++)
                if ((status = MPI_Recv(result[i*h+j], w, MPI_DOUBLE,
                                i, MPI_ANY_TAG, MPI_COMM_WORLD,
                                MPI_STATUS_IGNORE)))
                {
                    MPI_perror(status, "MPI_Recv");
                    return -1;
                }

        tend = get_clock();
        ttotal = tend-tstart;

        /* Results */
        /*
        printf("Final (%dx%d) grid (%d/%d):\n", h, w, rank+1, size);
        print_grid(result,n,n);
        */
        printf("Results:\n");
        printf("Iterations=%d\n",iteration-1);
        printf("Tolerance=%12.10lf\n",maxdiff);
        printf("Running time=%12.8lf\n",ttotal);
        printf("Value at (%d,%d)=%12.8lf\n",r,c,result[r-1][c-1]);
        free_matrix(result, n);
    }

    free_matrix(a, h+2);
    free_matrix(b, h+2);
    MPI_Finalize();

    return 0;
}
