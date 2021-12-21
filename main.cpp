#include <iostream>
#include <iomanip>
#include <vector>
#include <CL/sycl.hpp>

#include "sequences.cpp"

#define MATCH 1
#define MISMATCH -3
#define OPEN_GAP 5
#define EXTEND_GAP 2

//using namespace std;
using namespace sycl;

int sequential_sw(char *a, char *b,int m, int n) {

    std::vector<int> ha(m,0), hb(m,0), hc(m,0), ea(m,0), eb(m,0), fa(m,0), fb(m,0);
    std::vector<int> *h1, *h2, *h3, *e1, *e2, *f1, *f2;

    int srow = 0, erow = 0, scol = 0, ecol = 0,d_size = 0, maxscore = 0;

    for(int d=0; d<(m+n-1); d++) {

        // cycle the buffers
        switch (d%3)
        {
            case 0:
                h1 = &ha;
                h2 = &hb;
                h3 = &hc;
                break;

            case 1:
                h1 = &hc;
                h2 = &ha;
                h3 = &hb;
                break;

            case 2:
                h1 = &hb;
                h2 = &hc;
                h3 = &ha;           
                break;
            
            default:
                std::cout << "error in index mod 3\n";
                break;
        }
        switch (d%2)
        {
            case 0:
                e1 = &ea;
                e2 = &eb;
                f1 = &fa;
                f2 = &fb;
                break;

            case 1:
                e1 = &eb;
                e2 = &ea;
                f1 = &fb;
                f2 = &fa;
                break;
            
            default:
                std::cout << "error in index mod 2\n";
                break;
        }

        srow = max(0,d-n+1);
        erow = min(m-1,d);
        scol = d - erow;
        ecol = d -srow;
        d_size = erow - srow + 1;

        if (d<n) {

            for(int i = 1; i<d_size; i++) {

                // calculate and store e
                (*e1)[i] = max((*e2)[i-1], (*h2)[i-1] - OPEN_GAP) - EXTEND_GAP;
                // calculate and store f
                (*f1)[i] = max((*f2)[i], (*h2)[i] - OPEN_GAP) - EXTEND_GAP;
                // calculate and store h
                int h = 0;
                h = max(h,(*e1)[i]);
                h = max(h,(*f1)[i]);
                h = max(h,(*h3)[i-1]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                (*h1)[i] = h;
            }

        }
        else {
            if(d==n) {

                for(int i = 1; i<d_size; i++) {

                    // calculate and store e
                    (*e1)[i] = max((*e2)[i], (*h2)[i] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store f
                    (*f1)[i] = max((*f2)[i+1], (*h2)[i+1] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store h
                    int h = 0;
                    h = max(h,(*e1)[i]);
                    h = max(h,(*f1)[i]);
                    h = max(h,(*h3)[i]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                    (*h1)[i] = h;
                }
            }

            else {
                for(int i = 1; i<d_size; i++) {

                    // calculate and store e
                    (*e1)[i] = max((*e2)[i], (*h2)[i] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store f
                    (*f1)[i] = max((*f2)[i+1], (*h2)[i+1] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store h
                    int h = 0;
                    h = max(h,(*e1)[i]);
                    h = max(h,(*f1)[i]);
                    h = max(h,(*h3)[i+1]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                    (*h1)[i] = h;
                }
            }

        }

        for(int i = 1; i<d_size; i++) {
            maxscore = max(maxscore, (*h1)[i]);
        }
    }

    return maxscore;
}

int parallel_sw(queue &Q, char *a, char *b,int m, int n) {

    std::vector<int> ha(m,0), hb(m,0), hc(m,0), ea(m,0), eb(m,0), fa(m,0), fb(m,0);
    buffer bha{ha}, bhb{hb}, bhc{hc}, bea{ea}, beb{eb}, bfa{fa}, bfb{fb};
    std::vector<int> *h1, *h2, *h3, *e1, *e2, *f1, *f2;

    int srow = 0, erow = 0, scol = 0, ecol = 0,d_size = 0, maxscore = 0;

    for(int d=0; d<(m+n-1); d++) {

        srow = max(0,d-n+1);
        erow = min(m-1,d);
        scol = d - erow;
        ecol = d -srow;
        d_size = erow - srow + 1;

        if (d<n) {

            Q.submit([&](handler &h){
                // cycle the buffers
                switch (d%3)
                {
                    case 0:
                        accessor h1{bha,h};
                        accessor h2{bhb,h};
                        accessor h3{bhc,h};
                        break;

                    case 1:
                        accessor h1{bhc,h};
                        accessor h2{bha,h};
                        accessor h3{bhb,h};
                        break;

                    case 2:
                        accessor h1{bhb,h};
                        accessor h2{bhc,h};
                        accessor h3{bha,h};       
                        break;
                }
                switch (d%2)
                {
                    case 0:
                        accessor e1{bea,h};
                        accessor e2{beb,h};
                        accessor f1{bfa,h};
                        accessor f2{bfb,h};
                        break;

                    case 1:
                        accessor e1{beb,h};
                        accessor e2{bea,h};
                        accessor f1{bfb,h};
                        accessor f2{bfa,h};
                        break;
                }

                h.parallel_for(d_size-1, [=](id<1> index) {
                    int i = index + 1;
                    // calculate and store e
                    e1[i] = max(e2[i-1], h2[i-1] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store f
                    f1[i] = max(f2[i], h2[i] - OPEN_GAP) - EXTEND_GAP;
                    // calculate and store h
                    int h_i = 0;
                    h_i = max(h_i,e1[i]);
                    h_i = max(h_i,f1[i]);
                    h_i = max(h_i,h3[i-1]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                    h1[i] = h_i;
                });
            });

        }
        else {
            if(d==n) {

                Q.submit([&](handler &h){
                    // cycle the buffers
                    switch (d%3)
                    {
                        case 0:
                            accessor h1{bha,h};
                            accessor h2{bhb,h};
                            accessor h3{bhc,h};
                            break;

                        case 1:
                            accessor h1{bhc,h};
                            accessor h2{bha,h};
                            accessor h3{bhb,h};
                            break;

                        case 2:
                            accessor h1{bhb,h};
                            accessor h2{bhc,h};
                            accessor h3{bha,h};       
                            break;
                    }
                    switch (d%2)
                    {
                        case 0:
                            accessor e1{bea,h};
                            accessor e2{beb,h};
                            accessor f1{bfa,h};
                            accessor f2{bfb,h};
                            break;

                        case 1:
                            accessor e1{beb,h};
                            accessor e2{bea,h};
                            accessor f1{bfb,h};
                            accessor f2{bfa,h};
                            break;
                    }
                    h.parallel_for(d_size-1, [=](id<1> index) {
                        int i = index + 1;
                        // calculate and store e
                        e1[i] = max(e2[i], h2[i] - OPEN_GAP) - EXTEND_GAP;
                        // calculate and store f
                        f1[i] = max(f2[i+1], h2[i+1] - OPEN_GAP) - EXTEND_GAP;
                        // calculate and store h
                        int h = 0;
                        h_i = max(h_i,e1[i]);
                        h_i = max(h_i,f1[i]);
                        h_i = max(h_i,h3[i]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                        h1[i] = h_i;
                    });
                });
            }

            else {
                    Q.submit([&](handler &h){
                        // cycle the buffers
                        switch (d%3)
                        {
                            case 0:
                                accessor h1{bha,h};
                                accessor h2{bhb,h};
                                accessor h3{bhc,h};
                                break;

                            case 1:
                                accessor h1{bhc,h};
                                accessor h2{bha,h};
                                accessor h3{bhb,h};
                                break;

                            case 2:
                                accessor h1{bhb,h};
                                accessor h2{bhc,h};
                                accessor h3{bha,h};       
                                break;
                        }
                        switch (d%2)
                        {
                            case 0:
                                accessor e1{bea,h};
                                accessor e2{beb,h};
                                accessor f1{bfa,h};
                                accessor f2{bfb,h};
                                break;

                            case 1:
                                accessor e1{beb,h};
                                accessor e2{bea,h};
                                accessor f1{bfb,h};
                                accessor f2{bfa,h};
                                break;
                        }
                        h.parallel_for(d_size-1, [=](id<1> index) {
                            int i = index + 1;
                            // calculate and store e
                            e1[i] = max(e2[i], h2[i] - OPEN_GAP) - EXTEND_GAP;
                            // calculate and store f
                            f1[i] = max(f2[i+1], h2[i+1] - OPEN_GAP) - EXTEND_GAP;
                            // calculate and store h
                            int h_i = 0;
                            h_i = max(h_i,e1[i]);
                            h_i = max(h_i,f1[i]);
                            h_i = max(h_i,h3[i+1]+((a[srow+i]==b[ecol-i]) ? MATCH : MISMATCH));
                            h1[i] = h_i;
                        });
                    });
            }

        }

        switch (d%3)
        {
            case 0:
                host_accessor h1{bha,h};
                break;

            case 1:
                host_accessor h1{bhc,h};
                break;

            case 2:
                host_accessor h1{bhb,h};      
                break;
        }

        for(int i = 1; i<d_size; i++) {
            maxscore = max(maxscore, h1[i]);
        }
    }

    return maxscore;
}

int main(int argc, char *argv[])
{
    if(argc != 3) {
        std::cout << "incorrect usage\n";
        return 1;
    }

	char *a_header, *b_header, *a, *b;
	int m, n, ext_m, ext_n;

    // Load query sequence
	load_sequence(argv[1],&a,&a_header,&m,&ext_m,0);

	// Load target sequence
	load_sequence(argv[2],&b,&b_header,&n,&ext_n,0);

    // test
    std::cout << a_header << "\n";
    std::cout << "number of nucleotides: "<< m << "\n";
    for(int i=0; i<10; i++) {
        std::cout << a[i];
    }
    std::cout << "\n";
    std::cout << b_header << "\n";
    std::cout << "number of nucleotides: "<< n << "\n";
    for(int i=0; i<10; i++) {
        std::cout << b[i];
    }
    std::cout << "\n";

    time_t start, end;

    // sequential
    time(&start);
    int score = sequential_sw(a, b, m, n);
    time(&end);

    std::cout << "sequential score: " << score << "\n";

    double time_taken = double(end - start);
    std::cout << "Time taken by sequential program is : " << fixed
         << time_taken << setprecision(5);
    std::cout << " sec " << endl;

    // parallel
    queue Q{property::queue::in_order()};

    time(&start);
    int score = parallel_sw(&Q,a, b, m, n);
    time(&end);

    std::cout << "parallel score: " << score << "\n";

    double time_taken = double(end - start);
    std::cout << "Time taken by parallel program is : " << fixed
         << time_taken << setprecision(5);
    std::cout << " sec " << endl;

    return 0;
}
