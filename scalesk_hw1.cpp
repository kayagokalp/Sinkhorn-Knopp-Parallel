#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <cstdlib>

int parallel_sk(int *xadj, int *adj, int *txadj, int *tadj,
                double *rv, double *cv, int nov, int siter)
{

  double start = omp_get_wtime();
  for (int it = 0; it < siter; it++)
  {

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < nov - 1; i++)
    {
      double rsum = 0;
      int rowIndex = xadj[i];
      for (; rowIndex < xadj[i + 1]; rowIndex++)
      {
        rsum += cv[adj[rowIndex]];
      }
      if (rsum != 0)
      {
        rv[i] = 1.0 / rsum;
      }
    }

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < nov - 1; i++)
    {
      double csum = 0;
      int colIndex = txadj[i];
      for (; colIndex < txadj[i + 1]; colIndex++)
      {
        csum += rv[tadj[colIndex]];
      }
      if (csum != 0)
      {
        cv[i] = 1.0 / csum;
      }
    }

    double maxErrorSum = 0;
#pragma omp parallel for schedule(guided) reduction(max \
                                                    : maxErrorSum)
    for (int i = 0; i < nov - 1; i++)
    {
      double errorSum = 0;
      double errorValue = 0;
      int rowIndex = xadj[i];
      for (; rowIndex < xadj[i + 1]; rowIndex++)
      {
        errorSum += cv[adj[rowIndex]] * rv[i];
        errorValue = abs(1.0 - errorSum);
      }
      if (errorValue > maxErrorSum)
      {
        maxErrorSum = errorValue;
      }
    }
    std::cout << "iter " << it << " - error " << (maxErrorSum) << "\n";
  }

  double end = omp_get_wtime();
  std::cout << omp_get_max_threads() << " Threads  --  "
            << "Time: " << end - start << " s.\n";
  return 1;
}

void read_mtxbin(std::string bin_name, int number_of_iter)
{

  const char *fname = bin_name.c_str();
  FILE *bp;
  bp = fopen(fname, "rb");

  int *nov = new int;
  int *nnz = new int;

  fread(nov, sizeof(int), 1, bp);
  fread(nnz, sizeof(int), 1, bp);

  int *adj = new int[*nnz];
  int *xadj = new int[*nov];
  int *tadj = new int[*nnz];
  int *txadj = new int[*nov];

  fread(adj, sizeof(int), *nnz, bp);
  fread(xadj, sizeof(int), *nov + 1, bp);

  fread(tadj, sizeof(int), *nnz, bp);
  fread(txadj, sizeof(int), *nov + 1, bp);

  int inov = *nov + 1;

  double *rv = new double[inov];
  double *cv = new double[inov];

  for (int i = 0; i < inov; i++)
  {
    rv[i] = 1;
    cv[i] = 1;
  }

  parallel_sk(xadj, adj, txadj, tadj, rv, cv, *nov, number_of_iter); //or no_col
}

int main(int argc, char *argv[])
{
  if (argc > 0)
  {
    char *fname = argv[1];
    int number_of_thread = atoi(argv[3]);
    int number_of_iter = atoi(argv[2]);
    omp_set_dynamic(0);
    omp_set_num_threads(number_of_thread);
    read_mtxbin(fname, number_of_iter);
  }
  return 0;
}