#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void bubbleSort(int arr[], int n)
{
  int i, j, temp;
  for (i = 0; i < n - 1; i++)
  {
    for (j = 0; j < n - i - 1; j++)
    {
      if (arr[j] > arr[j + 1])
      {
        temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
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
  bubbleSort(arr, n);
  end_time = omp_get_wtime();
  double sequential_time = end_time - start_time;

  // Parallel Execution
  start_time = omp_get_wtime();
#pragma omp parallel
  {
    bubbleSort(arr, n);
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
