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
#define bool2str(b) (b)?"true":"false"

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
	    count,
	    bucket_size = 0,
	    rank;
	double t1, t2, t3;
	uint64_t *splitters, *elmnts, *sample,
		 *local_elmnts, *local_sample, *bucket,
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
	splitters	= (uint64_t *) malloc (sizeof (uint64_t) * nbuckets);
	elmnts		= (uint64_t *) malloc (sizeof (uint64_t) * size);
	sample		= (uint64_t *) malloc (sizeof (uint64_t) * (nbuckets-1)*nbuckets);
//	local_elmnts	= (uint64_t *) malloc (sizeof (uint64_t) * bsize);
	local_sample	= (uint64_t *) malloc (sizeof (uint64_t) * (nbuckets-1));
	//the size of each bucket is guaranteed to be less than
	//2*size/nbuckets becuase of the way we choose the sample
	bucket		= (uint64_t *) malloc (sizeof (uint64_t) * 2*bsize);

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
	MPI_Bcast(elmnts, size, MPI_UINT64_T, 0, MPI_COMM_WORLD);
	local_elmnts = elmnts + rank*bsize;


	/*
	 * LOCAL SAMPLE SELECT
	 */
	qsort(local_elmnts, bsize, sizeof (uint64_t), compare);
	for(j = 0; j < nbuckets - 1; j++) {
		local_sample[j] = local_elmnts[bsize/nbuckets*(j+1)];
	}
		/*
	for (j = 0; j < nbuckets; j++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (j != rank) continue;
		printf("[%d] My block starts at index %d, and I choose these sample members: ", rank, local_elmnts - elmnts);
		for (i = 0; i < nbuckets - 1; i++)
			printf("%d, ", local_sample[i]);
		printf("\n");
		for (i = 0; i < bsize; i++) {
			printf("[%d] %llu\n", rank, local_elmnts[i]);
		}
	}
		*/
	/*
	 * GLOBAL SAMPLE SELECT GATHER
	 */
	MPI_Allgather(local_sample, nbuckets-1, MPI_UINT64_T,
			sample, nbuckets-1, MPI_UINT64_T,
			MPI_COMM_WORLD);

	/*
	for (j = 0; j < nbuckets; j++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (j != rank) continue;
		printf("[%d] Here are all the samples: ", rank);
		for (i = 0; i < nbuckets*(nbuckets-1); i++) {
			printf("%d, ", sample[i]);
		}
		printf("and I sent ");
		for (i = 0; i < nbuckets - 1; i++)
			printf("%d, ", local_sample[i]);
		printf("\n");
	}
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		t3 = get_clock();
		printf("Sample Select Time: %lf\n",(t3-t1));
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
		/*
		printf("Splitters: ");
		for (i = 0; i < nbuckets; i++) {
			printf("%llu, ", splitters[i]);
		}
		printf("chosen from this list: ");
		for (i = 0; i < (nbuckets-1)*nbuckets; i++) {
			printf("%llu, ", sample[i]);
		}
		printf("\n");
		*/
		t3 = get_clock();
		printf("Splitter Select Time: %lf\n",(t3-t2));
		t2 = t3;
	}


	/*
	 * LOCAL BUCKET DISTRIBUTE
	 */
	for(i = 0; i < size; i++) {
		for (j = 0; j <= rank; j++) {
			if(elmnts[i] < splitters[j]) {
				if (j == rank) {
					bucket[bucket_size++] = elmnts[i];
				}
				break;
			}
		}
	}
	/*
	for (j = 0; j < nbuckets; j++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (j != rank) continue;
		printf("[%d] Pivot: %llu; Bucket Size: %d\n", rank, splitters[rank], bucket_size);
		for (i = 0; i < bucket_size; i++)
			printf("%llu, ", bucket[i]);
		printf("\n");
	}
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		t3 = get_clock();
		printf("Bucket Distribute Time: %lf\n",(t3-t2));
		t2 = t3;
	}


	/*
	 * LOCAL BUCKET SORT
	 */
	qsort(bucket, bucket_size, sizeof (uint64_t), compare);

	/*
	for (j = 0; j < nbuckets; j++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (j != rank) continue;
		printf("[%d] (sorted) Pivot: %llu; Bucket Size: %d\n", rank, splitters[rank], bucket_size);
		for (i = 0; i < bucket_size; i++)
			printf("%llu, ", bucket[i]);
		printf("\n");
	}
	*/

	MPI_Barrier(MPI_COMM_WORLD);
	if (!rank) {
		t3 = get_clock();
		printf("Bucket Sort Time: %lf\n",(t3-t2));
		printf("Total Time: %lf\n",(t3-t1));
	}


	#if CHECK
	{
		uint64_t local_check = 0, gather_check[nbuckets],
			 min_max[nbuckets][2], local_min_max[2];
		for(j = 0; j < bucket_size; j++) {
			local_check ^= bucket[j];
		}
		MPI_Gather(&local_check, 1, MPI_UINT64_T,
				gather_check, 1, MPI_UINT64_T,
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
		MPI_Gather(local_min_max, 2, MPI_UINT64_T,
				min_max, 2*nbuckets, MPI_UINT64_T,
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
	//free(local_elmnts);
	free(sample);
	free(local_sample);
	free(bucket);

	MPI_Finalize();
	return 0;
}
