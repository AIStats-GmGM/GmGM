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

SUBROUTINE SUM_LOG_SUM_2(X, Xs, Y, Ys, SL)
    INTEGER, INTENT(IN) :: Xs, Ys
    REAL, DIMENSION(Xs), INTENT(IN) :: X
    REAL, DIMENSION(Ys), INTENT(IN) :: Y
    REAL, INTENT(OUT) :: SL
    
    INTEGER :: IDX = 1
    REAL :: INTERMEDIATE = 1
    INTEGER :: SIMPLIFY_SIZE = 20
    
    DO I=1,Xs
        DO J=1,Ys
            INTERMEDIATE = INTERMEDIATE * (X(I)+Y(J))
            IF (IDX == SIMPLIFY_SIZE) THEN
                SL = SL + LOG(INTERMEDIATE)
                INTERMEDIATE = 1
                IDX = 1
            END IF
            IDX = IDX + 1
        END DO
    END DO
    SL = SL + LOG(INTERMEDIATE)
    RETURN
END SUBROUTINE