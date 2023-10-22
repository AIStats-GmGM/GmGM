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

SUBROUTINE PROJECT_INV_KRON_SUM_4(X, Xs, Y, Ys, Z, Zs, W, Ws, XProj, YProj, ZProj, WProj)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws
    REAL, INTENT(IN), DIMENSION(Xs) :: X
    REAL, INTENT(IN), DIMENSION(Ys) :: Y
    REAL, INTENT(IN), DIMENSION(Zs) :: Z
    REAL, INTENT(IN), DIMENSION(Ws) :: W
    REAL, INTENT(OUT), DIMENSION(Xs) :: XProj
    REAL, INTENT(OUT), DIMENSION(Ys) :: YProj
    REAL, INTENT(OUT), DIMENSION(Zs) :: ZProj
    REAL, INTENT(OUT), DIMENSION(Ws) :: WProj
    
    REAL, PARAMETER :: K_RATIO = 3./4.

    REAL :: CUR_VAL
    
    DO I=1,XS
        DO J=1,YS
            DO K=1,Zs
                DO L=1,Ws
                    CUR_VAL = 1 / (X(I)+Y(J)+Z(K)+W(L))
                    XProj(I) = XProj(I) + CUR_VAL
                    YProj(J) = YProj(J) + CUR_VAL
                    ZProj(K) = ZProj(K) + CUR_VAL
                    WProj(L) = WProj(L) + CUR_VAL
                END DO
            END DO
        END DO
    END DO

    ! Normalize
    XProj = XProj / (Ys*Zs*Ws)
    YProj = YProj / (Xs*Zs*Ws)
    ZProj = ZProj / (Xs*Ys*Ws)
    WProj = WProj / (Xs*Ys*Zs)

    ! Offset diagonal
    XProj = XProj - K_RATIO * SUM(XProj) / XS
    YProj = YProj - K_RATIO * SUM(YProj) / YS
    ZProj = ZProj - K_RATIO * SUM(ZProj) / ZS
    WProj = WProj - K_RATIO * SUM(WProj) / WS
END SUBROUTINE

SUBROUTINE PROJECT_INV_KRON_SUM_5(X, Xs, Y, Ys, Z, Zs, W, Ws, V, Vs, XProj, YProj, ZProj, WProj, VProj)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws, Vs
    REAL, INTENT(IN), DIMENSION(Xs) :: X
    REAL, INTENT(IN), DIMENSION(Ys) :: Y
    REAL, INTENT(IN), DIMENSION(Zs) :: Z
    REAL, INTENT(IN), DIMENSION(Ws) :: W
    REAL, INTENT(IN), DIMENSION(Vs) :: V
    REAL, INTENT(OUT), DIMENSION(Xs) :: XProj
    REAL, INTENT(OUT), DIMENSION(Ys) :: YProj
    REAL, INTENT(OUT), DIMENSION(Zs) :: ZProj
    REAL, INTENT(OUT), DIMENSION(Ws) :: WProj
    REAL, INTENT(OUT), DIMENSION(Vs) :: VProj
    
    REAL, PARAMETER :: K_RATIO = 4./5.

    REAL :: CUR_VAL
    
    DO I=1,XS
        DO J=1,YS
            DO K=1,Zs
                DO L=1,Ws
                    DO M=1,Vs
                        CUR_VAL = 1 / (X(I)+Y(J)+Z(K)+W(L)+V(M))
                        XProj(I) = XProj(I) + CUR_VAL
                        YProj(J) = YProj(J) + CUR_VAL
                        ZProj(K) = ZProj(K) + CUR_VAL
                        WProj(L) = WProj(L) + CUR_VAL
                        VProj(M) = VProj(M) + CUR_VAL
                    END DO
                END DO
            END DO
        END DO
    END DO

    ! Normalize
    XProj = XProj / (Ys*Zs*Ws*Vs)
    YProj = YProj / (Xs*Zs*Ws*Vs)
    ZProj = ZProj / (Xs*Ys*Ws*Vs)
    WProj = WProj / (Xs*Ys*Zs*Vs)
    VProj = VProj / (Xs*Ys*Zs*Ws)

    ! Offset diagonal
    XProj = XProj - K_RATIO * SUM(XProj) / XS
    YProj = YProj - K_RATIO * SUM(YProj) / YS
    ZProj = ZProj - K_RATIO * SUM(ZProj) / ZS
    WProj = WProj - K_RATIO * SUM(WProj) / WS
    VProj = VProj - K_RATIO * SUM(VProj) / VS
END SUBROUTINE

SUBROUTINE PROJECT_INV_KRON_SUM_6(X, Xs, Y, Ys, Z, Zs, W, Ws, V, Vs, U, Us, XProj, YProj, ZProj, WProj, VProj, UProj)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws, Vs, Us
    REAL, INTENT(IN), DIMENSION(Xs) :: X
    REAL, INTENT(IN), DIMENSION(Ys) :: Y
    REAL, INTENT(IN), DIMENSION(Zs) :: Z
    REAL, INTENT(IN), DIMENSION(Ws) :: W
    REAL, INTENT(IN), DIMENSION(Vs) :: V
    REAL, INTENT(IN), DIMENSION(Us) :: U
    REAL, INTENT(OUT), DIMENSION(Xs) :: XProj
    REAL, INTENT(OUT), DIMENSION(Ys) :: YProj
    REAL, INTENT(OUT), DIMENSION(Zs) :: ZProj
    REAL, INTENT(OUT), DIMENSION(Ws) :: WProj
    REAL, INTENT(OUT), DIMENSION(Vs) :: VProj
    REAL, INTENT(OUT), DIMENSION(Us) :: UProj
    
    REAL, PARAMETER :: K_RATIO = 5./6.

    REAL :: CUR_VAL
    
    DO I=1,XS
        DO J=1,YS
            DO K=1,Zs
                DO L=1,Ws
                    DO M=1,Vs
                        DO N=1,Us
                            CUR_VAL = 1 / (X(I)+Y(J)+Z(K)+W(L)+V(M)+U(N))
                            XProj(I) = XProj(I) + CUR_VAL
                            YProj(J) = YProj(J) + CUR_VAL
                            ZProj(K) = ZProj(K) + CUR_VAL
                            WProj(L) = WProj(L) + CUR_VAL
                            VProj(M) = VProj(M) + CUR_VAL
                            UProj(N) = UProj(N) + CUR_VAL
                        END DO
                    END DO
                END DO
            END DO
        END DO
    END DO

    ! Normalize
    XProj = XProj / (Ys*Zs*Ws*Vs*Us)
    YProj = YProj / (Xs*Zs*Ws*Vs*Us)
    ZProj = ZProj / (Xs*Ys*Ws*Vs*Us)
    WProj = WProj / (Xs*Ys*Zs*Vs*Us)
    VProj = VProj / (Xs*Ys*Zs*Ws*Us)
    UProj = UProj / (Xs*Ys*Zs*Ws*Vs)

    ! Offset diagonal
    XProj = XProj - K_RATIO * SUM(XProj) / XS
    YProj = YProj - K_RATIO * SUM(YProj) / YS
    ZProj = ZProj - K_RATIO * SUM(ZProj) / ZS
    WProj = WProj - K_RATIO * SUM(WProj) / WS
    VProj = VProj - K_RATIO * SUM(VProj) / VS
    UProj = UProj - K_RATIO * SUM(UProj) / US
END SUBROUTINE