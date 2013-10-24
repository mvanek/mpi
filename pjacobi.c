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

int
boundary_init (double **a, int hotlen, int coldlen, int n, int h, int w, int rank, int size)
{
    int i, j;
    int status;
    /* TOP */
    if (!rank) {            /* The top block */
        for (j = 0; j < w+2; j++)
            a[0][j] = a[1][j];
    } else {
        MPI_Recv(&a[0][1], w, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank != size-1)
        MPI_Send(&a[h][1], w, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);

    /* BOTTOM */
    if (rank == size-1) {   /* The bottom block */
        for (j = 0; j < coldlen; j++)
            a[h+1][j] = a[h][j];
    } else {
        MPI_Recv(&a[h+1][1], w, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank)
        MPI_Send(&a[1][1], w, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);

    /* LEFT */
    for (i = hotlen; i < h+2; i++)
        a[i][0] = a[i][1];

    /* RIGHT */
    for (i = 0; i < h+2; i++)
        a[i][n+1] = a[i][n];

    return 0;
}

double
compute_grid (double **a, double **b, int h, int w)
{
    int i, j;
    double maxdiff_loc = 0.0;
    for (i = 1; i < h+1; i++)
        for (j = 1; j < w+1; j++) {
            b[i][j] = 0.2*(a[i][j] + a[i-1][j] + a[i+1][j] +
                       a[i][j-1] + a[i][j+1]);
            if (fabs(b[i][j] - a[i][j]) > maxdiff_loc)
                maxdiff_loc = fabs(b[i][j] - a[i][j]);
        }
    return maxdiff_loc;
}

int
pjacobi (int n, int r, int c, int max_iterations, int rank, int size)
{
    int i, j;                       /* Runtime data */
    int bytes, status;
	double tstart, tend, ttotal;
    int iteration = 0;
    double **a, **b;                /* Matrix data */
    double maxdiff = 1.0;
    int h = n/size,
        w = n,
        hotlen = MAX(MIN(h, (n+2)/2 - h*rank), 0),
        coldlen = (n+2)/2;

    /* INIT */
	a = create_matrix(h+2, w+2);
	b = create_matrix(h+2, w+2);
	init_matrix(a,h+2, w+2, hotlen, coldlen, rank, size);

    matcpy(b, a, h, w);

    if (!rank) {
        printf("Running simulation with tolerance=%lf and max iterations=%d\n",
                TOL, max_iterations);
        tstart = get_clock();
    }

	while (TOL < maxdiff && iteration++ < max_iterations)
    {
        double maxdiff_loc;
        boundary_init(a, hotlen, coldlen, n, h, w, rank, size);
        maxdiff_loc = compute_grid(a, b, h, w);
		swap_matrix(&a,&b);
        MPI_Allreduce(&maxdiff_loc, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
	}
    if (rank) {
        for (i = 1; i < h+1; i++)
            MPI_Send(&a[i][1], w, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        double **result = create_matrix(n, n);
        matcpy(result, &a[1], h, w);
        for (i = 1; i < size; i++)
            for (j = 0; j < h; j++)
                MPI_Recv(result[i*h+j], w, MPI_DOUBLE, i, MPI_ANY_TAG,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        tend = get_clock();
        ttotal = tend-tstart;

        /* Results */
        printf("Results:\n");
        printf("Iterations=%d\n",iteration-1);
        printf("Tolerance=%12.10lf\n",maxdiff);
        printf("Running time=%12.8lf\n",ttotal);
        printf("Value at (%d,%d)=%12.8lf\n",r,c,result[r-1][c-1]);
        free_matrix(result, n);
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
