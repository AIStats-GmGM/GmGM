SUBROUTINE PROJECT_INV_KRON_SUM_2(X, Xs, Y, Ys, XProj, YProj)
    INTEGER, INTENT(IN) :: Xs, Ys
    REAL, INTENT(IN), DIMENSION(Xs) :: X
    REAL, INTENT(IN), DIMENSION(Ys) :: Y
    REAL, INTENT(OUT), DIMENSION(Xs) :: XProj
    REAL, INTENT(OUT), DIMENSION(Ys) :: YProj
    
    REAL, PARAMETER :: K_RATIO = 1./2.

    REAL :: CUR_VAL
    
    
    DO I=1,XS
        DO J=1,YS
            CUR_VAL = 1 / (X(I)+Y(J))
            XProj(I) = XProj(I) + CUR_VAL
            YProj(J) = YProj(J) + CUR_VAL
        END DO
    END DO

    ! Normalize
    XProj = XProj / Ys
    YProj = YProj / Xs

    ! Offset diagonal
    XProj = XProj - K_RATIO * SUM(XProj) / XS
    YProj = YProj - K_RATIO * SUM(YProj) / YS
    
END SUBROUTINE

SUBROUTINE PROJECT_INV_KRON_SUM_3(X, Xs, Y, Ys, Z, Zs, XProj, YProj, ZProj)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs
    REAL, INTENT(IN), DIMENSION(Xs) :: X
    REAL, INTENT(IN), DIMENSION(Ys) :: Y
    REAL, INTENT(IN), DIMENSION(Zs) :: Z
    REAL, INTENT(OUT), DIMENSION(Xs) :: XProj
    REAL, INTENT(OUT), DIMENSION(Ys) :: YProj
    REAL, INTENT(OUT), DIMENSION(Zs) :: ZProj
    
    REAL, PARAMETER :: K_RATIO = 2./3.

    REAL :: CUR_VAL
    
    DO I=1,XS
        DO J=1,YS
            DO K=1,Zs
                CUR_VAL = 1 / (X(I)+Y(J)+Z(K))
                XProj(I) = XProj(I) + CUR_VAL
                YProj(J) = YProj(J) + CUR_VAL
                ZProj(K) = ZProj(K) + CUR_VAL
            END DO
        END DO
    END DO

    ! Normalize
    XProj = XProj / (Ys*Zs)
    YProj = YProj / (Xs*Zs)
    ZProj = ZProj / (Xs*Ys)

    ! Offset diagonal
    XProj = XProj - K_RATIO * SUM(XProj) / XS
    YProj = YProj - K_RATIO * SUM(YProj) / YS
    ZProj = ZProj - K_RATIO * SUM(ZProj) / ZS
END SUBROUTINE