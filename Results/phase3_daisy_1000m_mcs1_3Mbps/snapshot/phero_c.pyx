# Cython functions

from libc.math cimport cos, sin, sqrt, INFINITY
import numpy as np
cimport numpy as np
import cython


cdef double pi = 3.14159265358979323846 

cdef (double,double,double) f_continuous(double theta, double v, double mu):
    cdef double xdot = v * sin(theta);
    cdef double ydot = v * cos(theta);
    cdef double thetadot = v * mu;
    return (xdot,ydot,thetadot);


cpdef (double,double,double) rk4_cy(double x, double y, double theta, double v, double mu, double dt):
    """ Runge-Kutta (rk4) """

    cdef double thetar = theta/180*pi;
    cdef double mur    = mu   /180*pi;
    cdef double k1_x, k1_y, k1_theta;
    cdef double k2_x, k2_y, k2_theta;
    cdef double k3_x, k3_y, k3_theta;
    cdef double k4_x, k4_y, k4_theta;
    k1_x, k1_y, k1_theta = f_continuous(thetar, v, mur);
    k2_x, k2_y, k2_theta = f_continuous(thetar + k1_theta*dt/2, v, mur);
    k3_x, k3_y, k3_theta = f_continuous(thetar + k2_theta*dt/2, v, mur);
    k4_x, k4_y, k4_theta = f_continuous(thetar + k3_theta*dt, v, mur);

    x += (k1_x+2*k2_x+2*k3_x+k4_x)*dt/6;
    y += (k1_y+2*k2_y+2*k3_y+k4_y)*dt/6;
    theta += 180/pi*(k1_theta+2*k2_theta+2*k3_theta+k4_theta)*dt/6;

    theta %= 360;
    return x,y,theta

cpdef float computeDistance(float x1, float y1, float x2, float y2):
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

cpdef float computeDistanceNoSqrt(float x1, float y1, float x2, float y2):
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)


from libcpp.vector cimport vector

cpdef vector[float] shortest_path_length_cy(np.ndarray[np.uint8_t, ndim=2] G, int source):    
    cdef:
        int n = G.shape[0]
        vector[float] distance = vector[float](n, INFINITY)
        vector[int] queue = vector[int](n)  # There won't ever be more than n pushes in a row
        int front = 0, back = 0
        int cur_node, nei

    # Init
    distance[source] = 0
    queue[back] = source
    back += 1

    # BFS
    while front < back:
        cur_node = queue[front]
        front += 1

        for nei in range(n):
            if G[cur_node, nei] and distance[nei] == INFINITY:
                distance[nei] = distance[cur_node] + 1
                queue[back] = nei
                back += 1

    return distance
    

@cython.boundscheck(False)
cpdef np.ndarray[np.uint8_t, ndim=2] warshall_cy(np.ndarray[np.uint8_t, ndim=2] G):
    cdef:
        int n = G.shape[0]
        int i, j, k
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                G[i, j] = G[i, j] | (G[i, k] & G[k, j])

    return G

from libcpp.algorithm cimport reverse

cpdef vector[int] shortest_path_cy(np.ndarray[np.uint8_t, ndim=2] G, int source, int target):
    cdef:
        int n = G.shape[0]
        vector[double] distance = vector[double](n, INFINITY)
        vector[int] predecessor = vector[int](n, -1)
        vector[int] queue = vector[int](n)
        int front = 0, back = 0
        int cur_node, nei

    # Init
    distance[source] = 0
    queue[back] = source
    back += 1

    # BFS
    while front < back:
        cur_node = queue[front]
        front += 1

        for nei in range(n):
            if G[cur_node, nei] and distance[nei] == INFINITY:
                distance[nei] = distance[cur_node] + 1
                predecessor[nei] = cur_node
                queue[back] = nei
                back += 1

                if nei == target:
                    front = back
                    break

    cdef vector[int] path
    cur_node = target
    while cur_node != -1:
        path.push_back(cur_node)
        cur_node = predecessor[cur_node]

    reverse(path.begin(), path.end())

    if path.size() == 0 or path[0] != source: # Check path exists (starts with source)
        return vector[int]()
    return path

