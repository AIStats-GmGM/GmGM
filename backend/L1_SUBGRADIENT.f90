! Actually slower than the python version

SUBROUTINE L1_SUBGRADIENT(EVALS, EVECS, S, RECON, SUBGRAD)
    INTEGER, INTENT(IN) :: S
    REAL, DIMENSION(S), INTENT(IN) :: EVALS
    REAL, DIMENSION(S, S), INTENT(IN) :: EVECS
    REAL, DIMENSION(S, S), INTENT(OUT) :: RECON
    REAL, DIMENSION(S), INTENT(OUT) :: SUBGRAD

    INTEGER :: I, J, K
    REAL :: CORE
    
    DO I = 1,S
        DO J = 1,S
            DO K = 1,S
                IF (I /= J) THEN
                    RECON(I, J) = RECON(I, J) + EVALS(K) * EVECS(I, K) * EVECS(J, K)
                END IF
            END DO
        END DO
    END DO

    DO I = 1,S
        DO J = 1,S
            IF (RECON(I, J) > 0) THEN
                CORE = 1
            ELSE IF (RECON(I, J) < 0) THEN
                CORE = -1
            ELSE
                CORE = 0
            END IF
            DO K = 1,S
                SUBGRAD(K) = SUBGRAD(K) + CORE * EVECS(I, K) * EVECS(J, K)
            END DO
        END DO
    END DO
END SUBROUTINE