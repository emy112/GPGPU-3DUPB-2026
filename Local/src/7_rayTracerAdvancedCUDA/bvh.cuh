#ifndef BVH_H
#define BVH_H

#include "hittable.cuh"
#include "hittable_list.cuh"

#include "ray.cuh"
#include "aabb.cuh"
#include "random.cuh"

#include <thrust/sort.h>

class bvh_node : public hittable {
    public:
        __device__ bvh_node() {}
        __device__ bvh_node(hittable *left, hittable * right) {
            _left = left;
            _right = right;
            _bbox = aabb(left->bounding_box(), right->bounding_box());
        }
        __device__ bvh_node(hittable_list *l, curandState *localState) : bvh_node(l->list, l->list_size, localState) {}
        __device__ bvh_node(hittable **l, int n, curandState *localState);
        __device__ virtual ~bvh_node() {
            if (_left == _right) {
                delete _left;
            } else {
                delete _left;
                delete _right;
            }
        }
        
        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const override;
        __device__ aabb bounding_box() const override;
    
    private:
        static __device__ bool box_compare(hittable *a,hittable *b, int axis_index) {
            aabb box_a = a->bounding_box();
            aabb box_b = b->bounding_box();
            
            return box_a.axis(axis_index).min < box_b.axis(axis_index).min;
        }

        static __device__ bool box_x_compare(hittable *a, hittable *b) {
            return box_compare(a, b, 0);
        }
        static __device__ bool box_y_compare(hittable *a, hittable *b) {
            return box_compare(a, b, 1);
        }
        static __device__ bool box_z_compare(hittable *a, hittable *b) {
            return box_compare(a, b, 2);
        }

        hittable *_left;
        hittable *_right;
        aabb _bbox;
        bool _last_level;
};

__device__ bool bvh_node::hit(const ray& r, interval ray_t, hit_record& rec) const {

    
    // TODO: Analyze and see the recursive version for traversal. Just build and run
    // the scene as-is. On a large enough tree, watch for it to crash with
    // an illegal memory access (device-side stack overflow)

    // After you figured it out, comment the recursive block below and write your OWN iterative version
    // here using an explicit stack, before peeking at the iterative reference implementation further down.

    //Recursive approach
    /* if (!_bbox.hit(r, ray_t))
         return false;

     bool hit_left = _left->hit(r, ray_t, rec);
     bool hit_right = _right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);

     return hit_left || hit_right;
     */
     // Iterative approach --------------------------------------------------------------------------------------------
     
    const bvh_node* stack[64];
    int stackTop = 0;
    bool hit = false;

     stack[stackTop++] = this;

     while (stackTop) {
         const bvh_node* currentNode = stack[--stackTop];

         if (!currentNode->_bbox.hit(r, ray_t))
             continue;

         if (currentNode->_last_level) {
             hit = currentNode->_left->hit(r, ray_t, rec);
             if (hit) {
                 return true;
             }
             hit = currentNode->_right->hit(r, ray_t, rec);
             if (hit) {
                 return true;
             }
         }
         else {
             stack[stackTop++] = (bvh_node*)currentNode->_left;
             stack[stackTop++] = (bvh_node*)currentNode->_right;
         }
     }
     return hit;
     
}

__device__ bvh_node::bvh_node(hittable **l, int n, curandState *localState) {

	// TODO: Analyze and see the recursive version for tree construction. Try to implement an iterative version here using an explicit stack, before peeking at

    // Recursive approach
     /*int axis = randomInt(localState, 0, 2);
     bool (*comparator)(hittable*, hittable*);
     switch (axis) {
         case 0:
             comparator = box_x_compare;
             break;
         case 1:
             comparator = box_y_compare;
             break;
         case 2:
             comparator = box_z_compare;
             break;
     }

     if (n == 1) {
         _left = _right = l[0];
     } else if (n == 2) {
         if (comparator(l[0], l[1])) {
             _left = l[0];
             _right = l[1];
         } else {
             _left = l[1];
             _right = l[0];
         }
     } else {
         thrust::sort(l, l + n, comparator);
         int mid = n / 2;
         _left = new bvh_node(l, mid, localState);
         _right = new bvh_node(l + mid, n - mid, localState);
     }

     _bbox = aabb(_left->bounding_box(), _right->bounding_box());
     */
    
    // Iterative approach -------------------------------------------------------------------------------------------
    
    // Get a modifiable copy of the list
    
    hittable** list = new hittable *[n];
    for (int i = 0; i < n; i++) {
       list[i] = l[i];
    }

    int levels = log2f(n);
    int nodes_per_level;

    // Sort sublists of objects along axis
    for (int level = 0; level < levels; level++) {
       nodes_per_level = (1 << level);
       int axis = randomInt(localState, 0, 2);
       bool (*comparator)(hittable*, hittable*);
       switch (axis) {
       case 0:
           comparator = box_x_compare;
           break;
       case 1:
           comparator = box_y_compare;
           break;
       case 2:
           comparator = box_z_compare;
           break;
       }

       for (int i = 0; i < n; i += n / nodes_per_level) {
           int left = i;
           int right = i + n / nodes_per_level;
           if (right > n)
               right = n;
           thrust::sort(list + left, list + right, comparator);
       }
    }

    bvh_node** nodes = new bvh_node *[n / 2];

    // Now we have the leaf nodes, we can build the tree - bottom up approach
    for (int i = 0; i < n / 2; i++) {
       nodes[i] = new bvh_node(list[i * 2], list[i * 2 + 1]);
       nodes[i]->_last_level = true;
    }

    n /= 2;

    while (n > 2) {
       for (int i = 0; i < n / 2; i++) {
           nodes[i] = new bvh_node(nodes[i * 2], nodes[i * 2 + 1]);
           nodes[i]->_last_level = false;
       }
       n /= 2;
    }

    _left = nodes[0];
    _right = nodes[1];
    _bbox = aabb(_left->bounding_box(), _right->bounding_box());

    // TODO: Have I forgotten something? Investigate memory leaks and fix them.
    delete[list];
    delete[nodes];
}

__device__ aabb bvh_node::bounding_box() const {
    return _bbox;
}


#endif
