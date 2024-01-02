from mpi4py import MPI
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def lu_decomposition(local_a, local_n):
    pivot_row = np.empty(n)
    for i in range(local_n):
        for j in range(size):
            if rank == j:
                #define the current index of pivot row
                pivot = i*size + j
                
                #pivot_row = np.empty( n )
                for k in range(pivot, n):
                    #store the elements in pivot row
                    pivot_row[k] = local_a[i,k]
                #broadcast to all other processors
                comm.Bcast([pivot_row, MPI.FLOAT], root=rank)
                
            else:
                pivot = i*size + j
                #pivot_row = np.empty( n )
                comm.Bcast([pivot_row, MPI.FLOAT], root=j)
                  
            if rank <= j:
                for k in range(i+1, local_n):
                    local_a[k, pivot] = local_a[k, pivot]/pivot_row[pivot]
                    for w in range(pivot+1, n):
                        local_a[k, w] = local_a[k, w] - pivot_row[w]*local_a[k, pivot]
                        
            if rank > j:
                for k in range(i, local_n):
                    local_a[k, pivot] = local_a[k, pivot]/pivot_row[pivot]
                    for w in range(pivot+1, n):
                        local_a[k, w] = local_a[k, w] - pivot_row[w]*local_a[k, pivot]
                        
    if rank == 0:
        for i in range(local_n):
            for j in range(n):
                A[i*size, j] = local_a[i, j]
                
    if rank != 0:
        for i in range(local_n):
            comm.Send([local_a[i, :], MPI.FLOAT], dest=0, tag=i)
    else:
        for i in range(1, size):
            for j in range(local_n):
                comm.Recv([local_a[j, :], MPI.FLOAT], source=i, tag=j)  
                for k in range(n):
                    A[j*size + i, k] = local_a[j,k]
                    
    if rank == 0:
        for i in range(n):
            for j in range(n):
                if i == j:
                    L[i,j] = 1.0
                    
                if i>j:
                    L[i,j] = A[i,j]
                else:
                    U[i,j] = A[i,j]
                             


def forward_backward_solve(L, U, b):
    
    # Forward substitution: Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i, j] * y[j]
            
    # Backward substitution: Ux = y       
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]

    return x


    
if __name__ == "__main__":
    
    start_time = MPI.Wtime() 
        
    #initialized n 
    n = 10000
                        
    if rank ==0:                
        # Create the matrix A and vector b
        A = np.full((n, n), 1.0)
        np.fill_diagonal(A, n)
        '''
        A=np.matrix([[1.,1.,0.,3.],
           [2.,1.,-1.,1.],
           [3.,-1.,-2.,2.],
           [-1.,2.,3.,-1.]])
        '''   
        #print(A)
        
    
    #Assume that the matrix size is a multiple of the number of CPUs used for execution. 
    #Set the number of rows for each processor as local_n
    local_n = n//size
    #Initialize the local matrix local_a and pivot row picot_row for p1 to p-1
    local_a = np.zeros((local_n, n), dtype=np.float64)
    #pivot_row = np.empty(n)
    
    #Initialize the L and U matrix for p0
    if rank == 0:
        U = np.zeros((n, n), dtype=np.float64)
        L = np.zeros((n, n), dtype=np.float64)
    
    #wrapped interleaved matrix a and send it accordingly to p1 - p-1
    if rank == 0:
        for i in range(local_n):
            for j in range(n):
                local_a[i,j] = A[i * size, j]

        for i in range(n):
            if i % size != 0:
                #set the dest processcor that this row should be send to
                dest_proc = i % size
                #set the tag for the dest processor which receive this row
                dest_tag = i // size + 1
                comm.Send([A[i, :], MPI.FLOAT], dest=dest_proc, tag=dest_tag)
    else:
        for i in range(local_n):
            #receive the ith row from processor 0 with the coresponding tag
            comm.Recv([local_a[i, :], MPI.FLOAT], source=0, tag=i+1)    
            
    #print(local_a)
    
    #lu decompositon
    lu_decomposition(local_a, local_n)
    
    end_time = MPI.Wtime()
    
    if rank == 0:
        print("For n = " + str(n))
        print("Time: " + str(end_time-start_time))
    
    '''
    if rank == 0:
        print(L)
        print(U)
    
        
    #forward_backward_solve
    if rank == 0:
        b = np.array([4, 1, -3, 4])
        #b = np.arange(1, n+1)
        ans = forward_backward_solve(L, U, b)
        print(ans)
    '''
    
    MPI.Finalize()
    