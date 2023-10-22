! Unlike the other Fortran files, this code is not a bottleneck,
! but by moving it to Fortran it is one less function to port
! to other API languages.

SUBROUTINE EXTRACT_D_VALUES(SHAPE_, NDIMS, D_ALL, D_LEFTS, D_RIGHTS)
    ! SHAPE represents the shape of a tensor,
    ! NDIMS is the number of dimensions, i.e. the
    ! length of SHAPE.  For matrices it is 2.
    INTEGER, INTENT(IN) :: NDIMS
    INTEGER, INTENT(IN), DIMENSION(NDIMS) :: SHAPE_

    ! D_ALL: (3, 4, 6) -> 3 * 4 * 6 = 72
    ! D_LEFTS: (3, 4, 6) -> (1, 3, 12)
    ! D_RIGHTS: (3, 4, 6) -> (24, 6, 1)
    INTEGER, INTENT(OUT) :: D_ALL
    INTEGER, INTENT(OUT), DIMENSION(NDIMS) :: D_LEFTS
    INTEGER, INTENT(OUT), DIMENSION(NDIMS) :: D_RIGHTS

    DO I = 1, NDIMS
        IF (I == 1) THEN
            D_LEFTS(I) = 1
            D_RIGHTS(I) = PRODUCT(SHAPE_(2:NDIMS))
        ELSE IF (I == NDIMS) THEN
            D_LEFTS(I) = PRODUCT(SHAPE_(1:I-1))
            D_RIGHTS(I) = 1
        ELSE
            D_LEFTS(I) = PRODUCT(SHAPE_(1:I-1))
            D_RIGHTS(I) = PRODUCT(SHAPE_(I+1:NDIMS))
        END IF
    END DO

    D_ALL = PRODUCT(SHAPE_)
END SUBROUTINE