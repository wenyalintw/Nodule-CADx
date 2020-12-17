#include <TH/TH.h>
#include <math.h>

int cpu_overlap(THFloatTensor * boxes1, THFloatTensor * boxes2, THFloatTensor * overlap) {
    THArgCheck(THFloatTensor_isContiguous(overlap), 3, "overlap must be contiguous");
    THArgCheck(THFloatTensor_isContiguous(boxes1), 1, "boxes1 must be contiguous");
    THArgCheck(THFloatTensor_isContiguous(boxes2), 2, "boxes2 must be contiguous");

    // Number of ROIs
    long boxes1_num = THFloatTensor_size(boxes1, 0);
    long boxes1_dim = THFloatTensor_size(boxes1, 1);

    long boxes2_num = THFloatTensor_size(boxes2, 0);
    long boxes2_dim = THFloatTensor_size(boxes2, 1);

    long overlap_num = THFloatTensor_size(overlap, 0);
    long overlap_dim = THFloatTensor_size(overlap, 1);

    float * boxes1_flat = THFloatTensor_data(boxes1);
    float * boxes2_flat = THFloatTensor_data(boxes2);
    float * overlap_flat = THFloatTensor_data(overlap);


    // nominal indices
    int i, j;
    //First box
    float ix0, iy0, iz0, id, ih, iw;
    float ix1, iy1, iz1;
    //Second box
    float xx0, yy0, zz0, dd, hh, ww;
    float xx1, yy1, zz1;
    //Overlap width, height and depth
    float w, h, d; 
    //Box1 and box2 area, intersection area and IoU
    float S1, S2, inter, ovr;

    for (i=0; i < boxes1_num; ++i) {
        iz0 = boxes1_flat[i * boxes1_dim ];
        iy0 = boxes1_flat[i * boxes1_dim + 1];
        ix0 = boxes1_flat[i * boxes1_dim + 2];
        id = boxes1_flat[i * boxes1_dim + 3];
        ih = boxes1_flat[i * boxes1_dim + 4];
        iw = boxes1_flat[i * boxes1_dim + 5];
        iz0 -= id / 2; iz1 = iz0 + id;
        iy0 -= ih / 2; iy1 = iy0 + ih;
        ix0 -= iw / 2; ix1 = ix0 + iw;

        for (j = 0; j < boxes2_num; ++j) {
            zz0 = boxes2_flat[j * boxes2_dim];
            yy0 = boxes2_flat[j * boxes2_dim + 1];
            xx0 = boxes2_flat[j * boxes2_dim + 2];
            dd = boxes2_flat[j * boxes2_dim + 3];
            hh = boxes2_flat[j * boxes2_dim + 4];
            ww = boxes2_flat[j * boxes2_dim + 5];
            zz0 -= dd / 2; zz1 = zz0 + dd;
            yy0 -= hh / 2; yy1 = yy0 + hh;
            xx0 -= ww / 2; xx1 = xx0 + ww;

            zz0 = fmaxf(iz0, zz0);
            yy0 = fmaxf(iy0, yy0);
            xx0 = fmaxf(ix0, xx0);

            zz1 = fminf(iz1, zz1);
            yy1 = fminf(iy1, yy1);
            xx1 = fminf(ix1, xx1);

            w = fmaxf(0.0, xx1 - xx0);
            h = fmaxf(0.0, yy1 - yy0);
            d = fmaxf(0.0, zz1 - zz0);
            inter = w * h * d;
            S1 = id * ih * iw;
            S2 = dd * hh * ww;

            ovr = inter / (S1 + S2 - inter);
            
            // printf("iz0 %f, iy0 %f, ix0 %f, d %f\n", iz0, iy0, ix0, dd);
            // printf("iz1 %f, iy1 %f, ix1 %f\n", iz1, iy1, ix1);

            // printf("zz0 %f, yy0 %f, xx0 %f, id %f\n", zz0, yy0, xx0, id);
            // printf("zz1 %f, yy1 %f, xx1 %f\n", zz1, yy1, xx1);
            // printf("w %f, h %f, d %f\n", w, h, d);
            // printf("S1: %f, S2: %f, inter: %f\n", S1, S2, inter);
            // printf("(%d, %d): S1 %f, S2 %f, inter %f, ovr %f\n", i, j, S1, S2, inter, ovr);

            overlap_flat[i * overlap_dim + j] = ovr;
        }
    }
    return 1;
}