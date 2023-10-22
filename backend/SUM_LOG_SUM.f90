SUBROUTINE SUM_LOG_SUM_2(X, Xs, Y, Ys, SL)
    INTEGER, INTENT(IN) :: Xs, Ys
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    REAL :: SM = 0

    SL = 0
    
    DO I=1,Xs
        DO J=1,Ys
            SM = X(I)+Y(J)
            INTERMEDIATE = INTERMEDIATE * SM
            IF (IDX == SIMPLIFY_SIZE) THEN
                SL = SL + LOG(INTERMEDIATE)
                INTERMEDIATE = 1.0
                IDX = 1
            END IF
            IDX = IDX + 1
        END DO
    END DO

    SL = SL + LOG(INTERMEDIATE)
END SUBROUTINE

SUBROUTINE SUM_LOG_SUM_3(X, Xs, Y, Ys, Z, Zs, SL)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, DIMENSION(Zs), INTENT(IN) :: Z
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    
    DO I=1,Xs
        DO J=1,Ys
            DO K=1,Zs
                INTERMEDIATE = INTERMEDIATE * (X(I)+Y(J)+Z(K))
                IF (IDX == SIMPLIFY_SIZE) THEN
                    SL = SL + LOG(INTERMEDIATE)
                    INTERMEDIATE = 1
                    IDX = 1
                END IF
                IDX = IDX + 1
            END DO
        END DO
    END DO
    RETURN
END SUBROUTINE

SUBROUTINE SUM_LOG_SUM_4(X, Xs, Y, Ys, Z, Zs, W, Ws, SL)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, DIMENSION(Zs), INTENT(IN) :: Z
    REAL, DIMENSION(Ws), INTENT(IN) :: W
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    
    DO I=1,Xs
        DO J=1,Ys
            DO K=1,Zs
                DO L=1,Ws
                    INTERMEDIATE = INTERMEDIATE * (X(I)+Y(J)+Z(K)+W(L))
                    IF (IDX == SIMPLIFY_SIZE) THEN
                        SL = SL + LOG(INTERMEDIATE)
                        INTERMEDIATE = 1
                        IDX = 1
                    END IF
                    IDX = IDX + 1
                END DO
            END DO
        END DO
    END DO
    SL = SL + LOG(INTERMEDIATE)
END SUBROUTINE

SUBROUTINE SUM_LOG_SUM_5(X, Xs, Y, Ys, Z, Zs, W, Ws, V, Vs, SL)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws, Vs
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, DIMENSION(Zs), INTENT(IN) :: Z
    REAL, DIMENSION(Ws), INTENT(IN) :: W
    REAL, DIMENSION(Vs), INTENT(IN) :: V
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    
    DO I=1,Xs
        DO J=1,Ys
            DO K=1,Zs
                DO L=1,Ws
                    DO M=1,Vs
                        INTERMEDIATE = INTERMEDIATE * (X(I)+Y(J)+Z(K)+W(L)+V(M))
                        IF (IDX == SIMPLIFY_SIZE) THEN
                            SL = SL + LOG(INTERMEDIATE)
                            INTERMEDIATE = 1
                            IDX = 1
                        END IF
                        IDX = IDX + 1
                    END DO
                END DO
            END DO
        END DO
    END DO
    SL = SL + LOG(INTERMEDIATE)
END SUBROUTINE

SUBROUTINE SUM_LOG_SUM_6(X, Xs, Y, Ys, Z, Zs, W, Ws, V, Vs, U, Us, SL)
    INTEGER, INTENT(IN) :: Xs, Ys, Zs, Ws, Vs, Us
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, DIMENSION(Zs), INTENT(IN) :: Z
    REAL, DIMENSION(Ws), INTENT(IN) :: W
    REAL, DIMENSION(Vs), INTENT(IN) :: V
    REAL, DIMENSION(Us), INTENT(IN) :: U
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    
    DO I=1,Xs
        DO J=1,Ys
            DO K=1,Zs
                DO L=1,Ws
                    DO M=1,Vs
                        DO N=1,Us
                            INTERMEDIATE = INTERMEDIATE * (X(I)+Y(J)+Z(K)+W(L)+V(M)+U(N))
                            IF (IDX == SIMPLIFY_SIZE) THEN
                                SL = SL + LOG(INTERMEDIATE)
                                INTERMEDIATE = 1
                                IDX = 1
                            END IF
                            IDX = IDX + 1
                        END DO
                    END DO
                END DO
            END DO
        END DO
    END DO
    SL = SL + LOG(INTERMEDIATE)
END SUBROUTINE