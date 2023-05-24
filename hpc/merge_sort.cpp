#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void merge(int arr[], int left[], int left_size, int right[], int right_size)
{
  int i, j, k;
  i = j = k = 0;

  while (i < left_size && j < right_size)
  {
    if (left[i] <= right[j])
    {
      arr[k++] = left[i++];
    }
    else
    {
      arr[k++] = right[j++];
    }
  }

  while (i < left_size)
  {
    arr[k++] = left[i++];
  }

  while (j < right_size)
  {
    arr[k++] = right[j++];
  }
}

void mergeSort(int arr[], int n)
{
  if (n < 2)
  {
    return;
  }

  int mid = n / 2;
  int left[mid];
  int right[n - mid];

  for (int i = 0; i < mid; i++)
  {
    left[i] = arr[i];
  }

  for (int i = mid; i < n; i++)
  {
    right[i - mid] = arr[i];
  }

#pragma omp parallel sections
  {
#pragma omp section
    {
      mergeSort(left, mid);
    }

#pragma omp section
    {
      mergeSort(right, n - mid);
    }
  }

  merge(arr, left, mid, right, n - mid);
}

int main()
{
  int i, n;

  printf("Enter the number of elements: ");
  scanf("%d", &n);

  int arr[n];

  printf("Enter elements:\n");
  for (i = 0; i < n; i++)
  {
    scanf("%d", &arr[i]);
  }

  double start_time, end_time;

  // Sequential Execution
  start_time = omp_get_wtime();
  mergeSort(arr, n);
  end_time = omp_get_wtime();
  double sequential_time = end_time - start_time;

  // Parallel Execution
  start_time = omp_get_wtime();
#pragma omp parallel
  {
#pragma omp single
    mergeSort(arr, n);
  }
  end_time = omp_get_wtime();
  double parallel_time = end_time - start_time;

  printf("\nSorted array: ");
  for (i = 0; i < n; i++)
  {
    printf("%d ", arr[i]);
  }

  // Calculate speedup
  double speedup = sequential_time / parallel_time;

  printf("\n\nExecution time (Sequential): %.6f seconds", sequential_time);
  printf("\nExecution time (Parallel): %.6f seconds", parallel_time);
  printf("\nSpeedup: %.2f\n", speedup);

  return 0;
}
