#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 10

void print_matrix(const char *name, double *arr, int n, int k)
{
    printf("%s: \n", name);
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            printf("%.f, ", arr[i * k + j]);
        }
        printf("\n");
    }
}

void transpose(double *arr, int n)
{
    int buf;
    for (int i = 0; i < n; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            buf = arr[j * n + i];
            arr[j * n + i] = arr[i * n + j];
            arr[i * n + j] = buf;
        }
    }
}

int main(int argc, char *argv[])
{
    int i, j, proc_num, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int block_size = N / proc_num;

    double a[N][N];
    double a1[N][block_size];
    double b[N][block_size];
    double b1[N][N];
    double initial[N][N];
    double transposed[N][N];

    if (rank == 0)
    {
        printf("Transposing a %dx%d matrix, divided among %d processors\n", N, N, proc_num);
    }

    if (N % proc_num != 0)
    {
        if (rank == 0)
        {
            printf("Error: %d should be multiple of process num. Current process num value: %d\n", N, proc_num);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        for (i = 0; i < N; ++i)
        {
            for (j = 0; j < N; j++)
            {
                initial[i][j] = 1000 * i + j + block_size * rank;
            }
        }
        print_matrix("Initial", &initial[0][0], N, N);
    }

    MPI_Datatype col, coltype;
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
    MPI_Type_create_resized(col, 0, sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    MPI_Scatter(&initial[0][0], block_size, coltype, &a[0][0], block_size, coltype, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < block_size; ++j)
        {
            a1[i][j] = a[i][j];
        }
    }

    MPI_Alltoall(&a1[0][0],
                 block_size * block_size,
                 MPI_DOUBLE,
                 &b[0][0],
                 block_size * block_size,
                 MPI_DOUBLE,
                 MPI_COMM_WORLD);

    for (i = 0; i < proc_num; i++)
    {
        transpose(&b[i * block_size][0], block_size);
    }

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < block_size; ++j)
        {
            b1[i][j] = b[i][j];
        }
    }

    MPI_Gather(&b1[0][0], block_size, coltype, &transposed[0][0], block_size, coltype, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        print_matrix("Transposed", &transposed[0][0], N, N);
        printf("Transpose seems ok\n");
    }

    MPI_Finalize();
    return 0;
}