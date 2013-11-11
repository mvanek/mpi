#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include <mpi.h>
#define SEED 100
#define OUTPUT 1
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
	int i, j, k,
	    size, bsize, nbuckets,
	    count,
	    *bucket_sizes;
	double t1, t2, t3;
	uint64_t *splitters, *elmnts, *sample, **buckets,
		 check;
	bool checkMax;

	if (argc != 2) {
		fprintf(stderr,
			"Wrong number of arguments.\nUsage: %s N\n",
			argv[0]);
		return -1;
	}

	nbuckets = 12;
	size = atoi(argv[1]);
	bsize = size/nbuckets;
	splitters	= (uint64_t *)  malloc (sizeof (uint64_t)   * nbuckets);
	elmnts		= (uint64_t *)  malloc (sizeof (uint64_t)   * size);
	sample		= (uint64_t *)  malloc (sizeof (uint64_t)   * size);
	buckets		= (uint64_t **) malloc (sizeof (uint64_t *) * nbuckets);
	//the size of each bucket is guaranteed to be less than
	//2*size/nbuckets becuase of the way we choose the sample
	for(i = 0; i < nbuckets; i++) {
		buckets[i] = (uint64_t *) malloc(2*bsize * sizeof (uint64_t));
	}
	bucket_sizes = (int *) malloc(sizeof (int) * nbuckets);
	memset(bucket_sizes, 0, sizeof (int) * nbuckets);

	//Fill elmnts with random numbers
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

	t1 = get_clock();

	/*
	 * SAMPLE SELECT
	 */
	for(i = 0, k = 0; i < nbuckets; i++) {
		qsort(&elmnts[i*bsize], bsize, sizeof (uint64_t), compare);
		for(j = 0; j < nbuckets - 1; j++, k++) {
			sample[k] = elmnts[i*bsize + bsize/nbuckets*(j+1)];
		}
	}

	t3 = get_clock();
	printf("Sample Select Time: %lf\n",(t3-t1));
	t2 = t3;


	/*
	 * SPLITTER SELECT
	 */
	qsort(sample, nbuckets * (nbuckets - 1), sizeof (uint64_t), compare);
	for(i = 1; i < nbuckets; i++) {
		splitters[i-1] = sample[i*(nbuckets-1)];
	}
	splitters[nbuckets-1] = ULLONG_MAX;

	t3 = get_clock();
	printf("Splitter Select Time: %lf\n",(t3-t2));
	t2 = t3;


	/*
	 * BUCKET DISTRIBUTE
	 */
	for(i = 0; i < size; i++) {
		for (j = 0; j < nbuckets; j++) {
			if(elmnts[i] < splitters[j]) {
				buckets[j][bucket_sizes[j]] = elmnts[i];
				bucket_sizes[j]++;
				break;
			}
		}
	}

	t3 = get_clock();
	printf("Bucket Distribute Time: %lf\n",(t3-t2));
	t2 = t3;


	/*
	 * BUCKET SORT
	 */
	for(i = 0; i < nbuckets; i++) {
		qsort(buckets[i], bucket_sizes[i], sizeof (uint64_t), compare);
	}

	t3 = get_clock();
	printf("Bucket Sort Time: %lf\n",(t3-t2));
	printf("Total Time: %lf\n",(t3-t1));


	#if CHECK
	count = 0;
	for(i = 0; i < nbuckets; i++) {
		for(j = 0; j < bucket_sizes[i]; j++) {
			check ^= buckets[i][j];
		}
		count += bucket_sizes[i];
	}
	printf("The bitwise xor is %llu\n",check);
	checkMax = true;
	for(i = 0; i < nbuckets - 1; i++) {
		if(buckets[i][bucket_sizes[i]-1] > buckets[i+1][0]) {
			checkMax = false;
		}
	}
	printf("The max of each bucket is not greater than the min of the next:	%s\n",
		bool2str(checkMax));
	#endif
	#if OUTPUT
	count = 0;
	for(i = 0; i < nbuckets; i++) {
		for(j = 0; j < bucket_sizes[i]; j++) {
			elmnts[count+j] = buckets[i][j];
		}
		count += bucket_sizes[i];
	}
	for(i = 0; i < size; i++) {
		printf("%llu\n", elmnts[i]);
	}
	#endif

	free(splitters);
	free(elmnts);
	free(sample);
	free(bucket_sizes);
	for(i = 0; i < nbuckets; i++) {
		free(buckets[i]);
	}
	free(buckets);
	
	return 0;
}
