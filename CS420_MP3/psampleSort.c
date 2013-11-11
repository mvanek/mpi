#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include <mpi.h>
#define SEED 100
#define OUTPUT 0
#define CHECK 1
#define bool2str(b) ((b)?"true":"false")

typedef unsigned long long uint64_t;

//Sequential sampleSort.  
//Assume size is a multiple of nbuckets*nbuckets

double get_clock() {
   struct timeval tv;
   if (gettimeofday(&tv, NULL) < 0) perror("gettimeofday");
   return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

int compare(const void *num1, const void *num2) {
	uint64_t *n1 = (uint64_t*)num1,
		 *n2 = (uint64_t*)num2;
	return (*n1 > *n2) - (*n1 < *n2);
}

int main(int argc, char *argv[]) {
	int i, j,
	    size, bsize, nbuckets,
	    *bucket_sizes, *incoming_bucket_sizes, bucket_size,
	    count,
	    rank;
	double t1, t2, t3;
	uint64_t *splitters, *elmnts, *sample,
		 *local_elmnts, *local_sample,
		 **buckets, *bucket,
		 check;
	bool checkMax;

	if (argc != 2) {
		fprintf(stderr,
			"Wrong number of arguments.\nUsage: %s N\n",
			argv[0]);
		return -1;
	}

	/*
	 * MPI INIT
	 */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nbuckets);

	size = atoi(argv[1]);
	bsize = size/nbuckets;
	splitters	= (uint64_t *)  malloc(sizeof (uint64_t)   * nbuckets);
	elmnts		= (uint64_t *)  malloc(sizeof (uint64_t)   * size);
	local_elmnts	= (uint64_t *)  malloc(sizeof (uint64_t)   * bsize);
	sample		= (uint64_t *)  malloc(sizeof (uint64_t)   * (nbuckets-1)*nbuckets);
	local_sample	= (uint64_t *)  malloc(sizeof (uint64_t)   * (nbuckets-1));
	buckets		= (uint64_t **) malloc(sizeof (uint64_t *) * nbuckets);
	//the size of each bucket is guaranteed to be less than
	//2*size/nbuckets becuase of the way we choose the sample
	bucket		= (uint64_t *)  malloc(sizeof (uint64_t *) * 2*bsize);
	for(i=0;i<nbuckets;i++) {
		buckets[i] = (uint64_t *) malloc(sizeof (uint64_t)*2*bsize);
	}
	bucket_sizes		= (int *) malloc(sizeof (int) * nbuckets);
	incoming_bucket_sizes	= (int *) malloc(sizeof (int) * nbuckets);
	for(i=0;i<nbuckets;i++) {
		bucket_sizes[i] = 0;
		incoming_bucket_sizes[i] = 0;
	}

	/*
	 * MASTER: GENERATE DATA
	 */
	if (!rank) {
		srand(SEED);
		for(i = 0; i < size; i++) {
			elmnts[i] = rand()%100;
		}
		#if CHECK
		check = 0;
		for(i = 0; i < size; i++) {
			check ^= elmnts[i];
		}
		#endif
	}

	MPI_Barrier(MPI_COMM_WORLD);
	t1 = get_clock();


	/*
	 * DISTRIBUTE DATA
	 */
	MPI_Scatter(elmnts, size, MPI_UNSIGNED_LONG_LONG,
			local_elmnts, bsize, MPI_UNSIGNED_LONG_LONG,
			0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[+] Data Distribution Time: %lf\n", ((t3 = get_clock()) - t1));
		t2 = t3;
	}


	/*
	 * LOCAL SAMPLE SELECT
	 */
	qsort(local_elmnts, bsize, sizeof (uint64_t), compare);
	for(j = 0; j < nbuckets - 1; j++) {
		local_sample[j] = local_elmnts[bsize/nbuckets*(j+1)];
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[-] Local Sample Select Time: %lf\n", ((t3 = get_clock()) - t1));
		t2 = t3;
	}


	/*
	 * GLOBAL SAMPLE SELECT GATHER
	 */
	MPI_Allgather(local_sample, nbuckets-1, MPI_UNSIGNED_LONG_LONG,
			sample, nbuckets-1, MPI_UNSIGNED_LONG_LONG,
			MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[+] Sample Select Gather Time: %lf\n", ((t3 = get_clock()) - t2));
		t2 = t3;
	}


	/*
	 * LOCAL SPLITTER SELECT (each process does the exact same calculation)
	 */
	qsort(sample, nbuckets * (nbuckets - 1), sizeof (uint64_t), compare);
	for(i = 1; i < nbuckets; i++) {
		splitters[i-1] = sample[i*(nbuckets-1)];
	}
	splitters[nbuckets-1] = ULLONG_MAX;

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[-] Splitter Select Time: %lf\n", ((t3 = get_clock()) - t2));
		t2 = t3;
	}


	/*
	 * LOCAL BUCKET DISTRIBUTE
	 */
	/* Sort elements into the buckets */
	for (i=0;i<size;i++) {
		for (j=0;j<nbuckets;j++) {
			if (local_elmnts[i]<splitters[j]) {
				buckets[j][bucket_sizes[j]] = local_elmnts[i];
				bucket_sizes[j]++;
				break;
			}
		}
	}
	/* Tell each process how many elements we will be sending */
	int bucket_disp[nbuckets];
	for (i=0;i<nbuckets;i++) {
		MPI_Gather(&bucket_sizes[i], 1, MPI_INT,
				incoming_bucket_sizes, 1, MPI_INT,
				i, MPI_COMM_WORLD);
	}
	bucket_disp[0] = 0;
	for (i=1;i<nbuckets;i++) {
		bucket_disp[i] = bucket_disp[i-1] + incoming_bucket_sizes[i-1];
	}
	bucket_size = bucket_disp[nbuckets-1] + incoming_bucket_sizes[nbuckets-1];
	/* Send the buckets to the appropriate process */
	for (i=0;i<nbuckets;i++) {
		MPI_Gatherv(buckets[j], bucket_sizes[i], MPI_UNSIGNED_LONG_LONG,
				bucket, incoming_bucket_sizes, bucket_disp, MPI_UNSIGNED_LONG_LONG,
				i, MPI_COMM_WORLD);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[ ] Bucket Distribute Time: %lf\n", ((t3 = get_clock()) - t2));
		t2 = t3;
	}


	/*
	 * LOCAL BUCKET SORT
	 */
	qsort(bucket, bucket_size, sizeof (uint64_t), compare);

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		printf("[-] Bucket Sort Time: %lf\n", ((t3 = get_clock()) - t2));
		printf("[-] Total Time: %lf\n",(t3-t1));
	}


	#if CHECK
	{
		uint64_t local_check = 0, gather_check[nbuckets],
			 min_max[nbuckets][2], local_min_max[2];
		for(j = 0; j < bucket_size; j++) {
			local_check ^= bucket[j];
		}
		MPI_Gather(&local_check, 1, MPI_UNSIGNED_LONG_LONG,
				gather_check, 1, MPI_UNSIGNED_LONG_LONG,
				0, MPI_COMM_WORLD);
		MPI_Reduce(&bucket_size, &count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
		if (!rank) {
			for (i = 0; i < nbuckets; i++) {
				check ^= gather_check[i];
			}
			printf("The bitwise xor is %llu\n",check);
		}

		local_min_max[0] = bucket[0];
		local_min_max[1] = bucket[bucket_size-1];
		MPI_Gather(local_min_max, 2, MPI_UNSIGNED_LONG_LONG,
				min_max, 2*nbuckets, MPI_UNSIGNED_LONG_LONG,
				0, MPI_COMM_WORLD);
		if (!rank) {
			checkMax = true;
			for (i = 0; i < (nbuckets-1); i++) {
				if (min_max[2*i + 1] > min_max[2 * (i+1)]) {
					checkMax = false;
				}
			}
			printf("The max of each bucket is not greater than the min of the next:	%s\n",
				bool2str(checkMax));
		}
	}
	#endif
	#if OUTPUT
	{
		for(i = 0; i < nbuckets; i++) {
			MPI_Barrier(MPI_COMM_WORLD);
			if (i == rank) {
				for (j = 0; j < bucket_size; j++) {
					printf("%llu\n", bucket[j]);
				}
			}
		}
	}
	#endif

	free(splitters);
	free(elmnts);
	free(sample);
	free(local_elmnts);
	free(local_sample);
	free(bucket);

	MPI_Finalize();
	return 0;
}
