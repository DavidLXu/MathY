//混进了奇怪的C++代码
//自己封装复数类，来进行傅里叶变换
//当然也可以用std::complex<T>
//https://zhuanlan.zhihu.com/p/31584464
//傅里叶变换可以用来加速多项式乘法

//代码有了，还不知怎么调用

#include <iostream>
struct Complex  {
	double r, i ;
	Complex ( )  {	}
	Complex ( double r, double i ) : r ( r ), i ( i )  {	}
	inline void real ( const double& x )  {  r = x ;  }
	inline double real ( )  {  return r ;  }
	inline Complex operator + ( const Complex& rhs )  const  {
		return Complex ( r + rhs.r, i + rhs.i ) ;
	}
	inline Complex operator - ( const Complex& rhs )  const  {
		return Complex ( r - rhs.r, i - rhs.i ) ;
	}
	inline Complex operator * ( const Complex& rhs )  const  {
		return Complex ( r * rhs.r - i * rhs.i, r * rhs.i + i * rhs.r ) ;
	}
	inline void operator /= ( const double& x )   {
		r /= x, i /= x ;
	}
	inline void operator *= ( const Complex& rhs )   {
		*this = Complex ( r * rhs.r - i * rhs.i, r * rhs.i + i * rhs.r ) ;
	}
	inline void operator += ( const Complex& rhs )   {
		r += rhs.r, i += rhs.i ;
	}
	inline Complex conj ( )  {
		return Complex ( r, -i ) ;
	}
} ;



bool inverse = false ;

inline Complex omega ( const int& n, const int& k )  {
    if ( ! inverse ) return Complex ( cos ( 2 * PI / n * k ), sin ( 2 * PI / n * k ) ) ;
    return Complex ( cos ( 2 * PI / n * k ), sin ( 2 * PI / n * k ) ).conj ( ) ;
}

inline void fft ( Complex *a, const int& n )  {
    if ( n == 1 ) return ;

    static Complex buf [N] ;
    
    const int m = n >> 1 ;
    
    for ( int i = 0 ; i < m ; ++ i )  {
        buf [i] = a [i << 1] ;
        buf [i + m] = a [i << 1 | 1] ;
    }
    
    memcpy ( a, buf, sizeof ( Complex ) * ( n + 1 ) ) ;

    Complex *a1 = a, *a2 = a + m;
    fft ( a1, m ) ;
    fft ( a2, m ) ;

    for ( int i = 0 ; i < m ; ++ i )  {
        Complex t = omega ( n, i ) ;
        buf [i] = a1 [i] + t * a2 [i] ;
        buf [i + m] = a1 [i] - t * a2 [i] ;
    }
    
    memcpy ( a, buf, sizeof ( Complex ) * ( n + 1 ) ) ;
}


struct FastFourierTransform  {
    Complex omega [N], omegaInverse [N] ;

    void init ( const int& n )  {
        for ( int i = 0 ; i < n ; ++ i )  {
            omega [i] = Complex ( cos ( 2 * PI / n * i), sin ( 2 * PI / n * i ) ) ;
            omegaInverse [i] = omega [i].conj ( ) ;
        }
    }

    void transform ( Complex *a, const int& n, const Complex* omega ) {
        for ( int i = 0, j = 0 ; i < n ; ++ i )  {
		if ( i > j )  std :: swap ( a [i], a [j] ) ;
		for( int l = n >> 1 ; ( j ^= l ) < l ; l >>= 1 ) ;
	}

        for ( int l = 2 ; l <= n ; l <<= 1 )  {
            int m = l / 2;
            for ( Complex *p = a ; p != a + n ; p += l )  {
                for ( int i = 0 ; i < m ; ++ i )  {
                    Complex t = omega [n / l * i] * p [m + i] ;
                    p [m + i] = p [i] - t ;
                    p [i] += t ;
                }
            }
        }
    }

    void dft ( Complex *a, const int& n )  {
        transform ( a, n, omega ) ;
    }

    void idft ( Complex *a, const int& n )  {
        transform ( a, n, omegaInverse ) ;
        for ( int i = 0 ; i < n ; ++ i ) a [i] /= n ;
    }
} fft ;

int main()
{


}