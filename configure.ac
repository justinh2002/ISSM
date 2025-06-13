# =====================================================================
#  configure.ac — ISSM (compact version with embedded AD flags)
# =====================================================================

AC_INIT([Ice-sheet and Sea-level System Model (ISSM)],
        [4.24],
        [https://issm.ess.uci.edu/forum/],
        [issm],
        [http://issm.jpl.nasa.gov])

AC_CONFIG_AUX_DIR([aux-config])
AC_CONFIG_MACRO_DIR([m4])
m4_include([m4/issm_options.m4])

AC_CANONICAL_TARGET

# ------------------------  Compilers  ---------------------------------
AC_PROG_CC([mpicc icc cl icl gcc])
AC_PROG_CPP
AC_PROG_CXX([mpicxx mpic++ icpc cl icl g++])
AC_PROG_F77([mpifort ifort g77 gfortran])
AC_PROG_FC([mpifort ifort gfortran])

AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])
AM_PROG_AR
LT_INIT([win32-dll])

# =====================================================================
#  Automatic-Differentiation switches
# =====================================================================
AC_ARG_ENABLE([ad],
  AS_HELP_STRING([--enable-ad],
                 [Build ISSM with CoDiPack+MediPack (disables PETSc)]),
  [enable_ad=$enableval],
  [enable_ad=no])

AC_ARG_WITH([codipack-dir],
  AS_HELP_STRING([--with-codipack-dir=DIR],  [Prefix of CoDiPack install]),
  [CODIPACK_DIR=$withval],
  [CODIPACK_DIR=])

AC_ARG_WITH([medipack-dir],
  AS_HELP_STRING([--with-medipack-dir=DIR],  [Prefix of MediPack install]),
  [MEDIPACK_DIR=$withval],
  [MEDIPACK_DIR=])

# --- Validation & flag embedding -------------------------------------
if test "x$enable_ad" = "xyes"; then
  if test -z "$CODIPACK_DIR" || test -z "$MEDIPACK_DIR"; then
    AC_MSG_ERROR([--enable-ad needs BOTH --with-codipack-dir and --with-medipack-dir])
  fi

  # Tell source files we are in AD mode
  AC_DEFINE([ISSM_USE_AD], [1], [Define to 1 if building with automatic differentiation])

  # Drop PETSc automatically (picked up by m4/issm_options.m4)
  ENABLE_PETSC=no
  AM_CONDITIONAL([USE_AD], [true])
else
  ENABLE_PETSC=yes
  AM_CONDITIONAL([USE_AD], [false])
fi

dnl ---- Embed include-paths, lib-paths & libs globally -----------------
if test "x$enable_ad" = "xyes"; then
  AM_CPPFLAGS="$AM_CPPFLAGS -I$CODIPACK_DIR/include -I$MEDIPACK_DIR/include -DCODI_ForcedInlines"
  AM_LDFLAGS="$AM_LDFLAGS -L$CODIPACK_DIR/lib -L$MEDIPACK_DIR/lib"
  LIBS="$LIBS -lcodi -lmedi"
fi

dnl  Export the variables so Automake can substitute them
AC_SUBST([AM_CPPFLAGS])
AC_SUBST([AM_LDFLAGS])
AC_SUBST([LIBS])

# ---------------------------------------------------------------------
#  Run ISSM’s usual option-detection macro collection
# ---------------------------------------------------------------------
ISSM_OPTIONS

# ---------------------------------------------------------------------
#  Output files
# ---------------------------------------------------------------------
AC_CONFIG_FILES([
        Makefile
        src/Makefile
        src/c/Makefile
        src/m/Makefile
        src/wrappers/Makefile
        src/wrappers/python/Makefile
        src/wrappers/matlab/Makefile
        src/wrappers/javascript/Makefile
])

AC_OUTPUT
