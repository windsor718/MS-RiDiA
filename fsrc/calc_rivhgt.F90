program calc_rivwth
! ================================================
#ifdef UseCDF
USE NETCDF
#endif
      implicit none
! ===================
! calculation type
      character*256       ::  buf

      character*256       ::  type            !! 'bin' for binary, 'cdf' for netCDF
      data                    type /'bin'/

! parameter
      real                ::  HC, HP, HO, HMIN    !! Coef, Power, Offset, Minimum for Height (H=max(HMIN,HC*Qave**HP)

      data                    HC   /0.1/
      data                    HP   /0.5/
      data                    HO   /0.0/
      data                    HMIN /0.5/

! river network map parameters
      integer             ::  ix, iy
      integer             ::  nx, ny          !! river map grid number
      integer             ::  ilp
! river netwrok map
      integer,allocatable ::  nextx(:,:)      !! downstream x
      integer,allocatable ::  nexty(:,:)      !! downstream y
! variable
      real,allocatable    ::  rivout(:,:)     !! discharge     [m3/s]
      real,allocatable    ::  rivhgt(:,:)     !! channel depth [m]
! file
      character*256       ::  cnextxy, crivout, crivhgt
      integer             ::  ios
!
! Undefined Values
      integer             ::  imis                !! integer undefined value
      real                ::  rmis                !! real    undefined value
      parameter              (imis = -9999)
      parameter              (rmis = 1.e+20)
! ================================================
! read parameters from arguments

      call getarg(1,buf)
       if( buf/='' ) read(buf,'(a256)') crivout
      call getarg(2,buf)
       if( buf/='' ) read(buf,'(a256)') crivhgt
      call getarg(3,buf)
       if( buf/='' ) read(buf,'(a256)') cnextxy
      call getarg(4,buf)
       if( buf/='' ) read(buf,*) HC
      call getarg(5,buf)
       if( buf/='' ) read(buf,*) HP
      call getarg(6,buf)
       if( buf/='' ) read(buf,*) nx
      call getarg(7,buf)
       if( buf/='' ) read(buf,*) ny

      write(*,*) 'HEIGHT H=max(', HMIN, ',', HC, '*Qave**',HP, '+', HO, ')'

! ===============================

      allocate(nextx(nx,ny),nexty(nx,ny))
      allocate(rivout(nx,ny),rivhgt(nx,ny))

! ===================

      open(11,file=trim(cnextxy),form='unformatted',access='direct',recl=4*nx*ny,status='old',iostat=ios)
      read(11,rec=1) nextx
      read(11,rec=2) nexty
      close(11)

      open(13,file=trim(crivout),form='unformatted',access='direct',recl=4*nx*ny)
      read(13,rec=1) rivout
      close(13)

! =============================
      do iy=1, ny
        do ix=1, nx
          if( nextx(ix,iy)/=imis )then
            rivhgt(ix,iy)=max( HMIN,  HC  * rivout(ix,iy)**HP +HO )
          else
            rivhgt(ix,iy)=-9999
          endif
        end do
      end do
! =============================

      open(22,file=crivhgt,form='unformatted',access='direct',recl=4*nx*ny)
      write(22,rec=1) rivhgt
      close(22)
end program calc_rivwth
