program calc_rivbta
! ================================================
      implicit none
! ===================
! calculation type
      character*256       ::  buf

      character*256       ::  type            !! 'bin' for binary, 'cdf' for netCDF
      data                    type /'bin'/


! parameter

      real                ::  b1g1, b1g2, b1g3, b1g4, b1g5, b1g6, b1g7, b1g8 !! beta1 regression parameters
      real                ::  b2g1, b2g2, b2g3, b2g4, b2g5, b2g6, b2g7, b2g8 !! beta2 regression parameters
      real                ::  b3g1, b3g2, b3g3, b3g4, b3g5, b3g6, b3g7, b3g8 !! beta3 regression parameters
      real                ::  b4g1, b4g2, b4g3, b4g4, b4g5, b4g6, b4g7, b4g8 !! beta4 regression parameters
      real                ::  sp  !! rivshp(ix,iy)

      data                    b1g1 /0.328523411998739/
      data                    b1g2 /-0.136629598093381/
      data                    b1g3 /0.226782890677302/
      data                    b1g4 /0.407317315366405/
      data                    b1g5 /0.0573936111784235/
      data                    b1g6 /-0.000827027398309836/
      data                    b1g7 /7.73363755705211e-06/
      data                    b1g8 /-0.218593761377831/

      data                    b2g1 /-1.73499845293949/
      data                    b2g2 /1.86961902684339/
      data                    b2g3 /-0.0787095731609086/
      data                    b2g4 /-1.70280956920105/
      data                    b2g5 /-0.236031863429058/
      data                    b2g6 /0.00114799994407101/
      data                    b2g7 /3.43885058357733e-06/
      data                    b2g8 /1.73575973862887/

      data                    b3g1 /-0.0914107976436178/
      data                    b3g2 /0.0400444500138610/
      data                    b3g3 /0.387741337558434/
      data                    b3g4 /-0.152454059221963/
      data                    b3g5 /-0.0859683978597510/
      data                    b3g6 /0.00121037617690324/
      data                    b3g7 /-1.24573143624190e-05/
      data                    b3g8 /0.515523126751616/

      data                    b4g1 /0.440010360908978/
      data                    b4g2 /-0.291924093276058/
      data                    b4g3 /0.00781944059584046/
      data                    b4g4 /0.0164084553274571/
      data                    b4g5 /0.0337017710333279/
      data                    b4g6 /-0.000464008615742078/
      data                    b4g7 /4.69370047950512e-06/
      data                    b4g8 /-0.204804996998889/

! river network map parameters
      integer             ::  ix, iy
      integer             ::  nx, ny          !! river map grid number
! river netwrok map
      integer,allocatable ::  nextx(:,:)      !! downstream x
      integer,allocatable ::  nexty(:,:)      !! downstream y
! variable
      real,allocatable    ::  rivshp(:,:)     !! channel cross section parameter
      real,allocatable    ::  rivbta(:,:,:)    !! river channel wetter perimeter parameters
! file
      character*256       ::  cnextxy, crivshp, crivbta
      integer             ::  ios
      integer             ::  ilp

! Undefined Values
      integer             ::  imis                !! integer undefined value
      parameter              (imis = -9999)
! ================================================
! read parameters from arguments

      call getarg(1,buf)
       if( buf/='' ) read(buf,'(a256)') crivshp
      call getarg(2,buf)
       if( buf/='' ) read(buf,'(a256)') crivbta
      call getarg(3,buf)
       if( buf/='' ) read(buf,'(a256)') cnextxy
      call getarg(4,buf)
       if( buf/='' ) read(buf,*) nx
      call getarg(5,buf)
       if( buf/='' ) read(buf,*) ny

! ==============================

      allocate(nextx(nx,ny))
      allocate(nexty(nx,ny))
      allocate(rivshp(nx,ny))
      allocate(rivbta(4,nx,ny))

! ===================

      open(11,file=trim(cnextxy),form='unformatted',access='direct',recl=4*nx*ny,status='old',iostat=ios)
      read(11,rec=1) nextx
      read(11,rec=2) nexty
      close(11)

      open(15,file=trim(crivshp),form='unformatted',access='direct',recl=4*nx*ny)
      read(15,rec=1) rivshp
      close(13)

! =============================
      do iy=1, ny
        do ix=1, nx
          if( nextx(ix,iy)/=imis )then
            sp = rivshp(ix,iy)
            rivbta(1,ix,iy)=b1g1+b1g2*sp**(-1.)+b1g3*sp**(-2.)+b1g4*sp**(-3)+b1g5*sp+b1g6*sp**(2.)+b1g7*sp**(3.)+b1g8*sp**(0.5)
            rivbta(2,ix,iy)=b2g1+b2g2*sp**(-1.)+b2g3*sp**(-2.)+b2g4*sp**(-3)+b2g5*sp+b2g6*sp**(2.)+b2g7*sp**(3.)+b2g8*sp**(0.5)
            rivbta(3,ix,iy)=b3g1+b3g2*sp**(-1.)+b3g3*sp**(-2.)+b3g4*sp**(-3)+b3g5*sp+b3g6*sp**(2.)+b3g7*sp**(3.)+b3g8*sp**(0.5)
            rivbta(4,ix,iy)=b4g1+b4g2*sp**(-1.)+b4g3*sp**(-2.)+b4g4*sp**(-3)+b4g5*sp+b4g6*sp**(2.)+b4g7*sp**(3.)+b4g8*sp**(0.5)
          else
            rivbta(:,ix,iy)=-9999
          endif
        end do
      end do
! =============================

      open(22,file=trim(crivbta),form='unformatted',access='direct',recl=4*nx*ny)
      do ilp=1, 4, 1
        write(22,rec=ilp) rivbta(ilp,:,:)
      end do
      close(22)

!!================================================

end program calc_rivbta
