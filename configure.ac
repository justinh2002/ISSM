# =====================================================================
#  configure.ac — Ice-sheet and Sea-Level System Model (ISSM)
#  Upstream:  https://issm.jpl.nasa.gov
#  Version :  4.24    –    updated 2025-06-12 for AD build flavour
# =====================================================================

AC_INIT([Ice-sheet and Sea-level System Model (ISSM)],
        [4.24],
        [https://issm.ess.uci.edu/forum/],
        [issm],
        [http://issm.jpl.nasa.gov])

# ---------------------------------------------------------------------
#  Auxiliary directories & m4 macros
# ---------------------------------------------------------------------
AC_CONFIG_AUX_DIR([aux-config])
AC_CONFIG_MACRO_DIR([m4])
m4_include([m4/issm_options.m4])

# ---------------------------------------------------------------------
#  Pretty banner
# ---------------------------------------------------------------------
AC_MSG_NOTICE([============================================================================])
AC_MSG_NOTICE([=           ${PACKAGE_NAME} ${PACKAGE_VERSION} configuration           =])
AC_MSG_NOTICE([============================================================================])

# ---------------------------------------------------------------------
#  System type / canonical target triple
# ---------------------------------------------------------------------
AC_CANONICAL_TARGET

# ---------------------------------------------------------------------
#  Compilers
# ---------------------------------------------------------------------
AC_PROG_CC([mpicc icc cl icl gcc])
AC_PROG_CPP
AC_PROG_CXX([mpicxx mpic++ icpc cl icl g++])
AC_PROG_F77([mpifort ifort g77 gfortran])
AC_PROG_FC([mpifort ifort gfortran])

# ---------------------------------------------------------------------
#  Automake / Libtool
# ---------------------------------------------------------------------
AM_INIT_AUTOMAKE([foreign])
AC_CONFIG_HEADERS([config.h])
AM_SILENT_RULES([yes])
AM_PROG_AR
LT_INIT([win32-dll])

# =====================================================================
#  Automatic-Differentiation (CoDiPack + MediPack) options
# =====================================================================
AC_ARG_ENABLE([ad],
  AS_HELP_STRING([--enable-ad],
                 [Build ISSM with CoDiPack+MediPack automatic differentiation (disables PETSc)]),
  [enable_ad=$enableval],
  [enable_ad=no])

AC_ARG_WITH([codipack-dir],
  AS_HELP_STRING([--with-codipack-dir=PREFIX],
                 [Prefix where CoDiPack is installed]),
  [CODIPACK_DIR=$withval],
  [CODIPACK_DIR=])

AC_ARG_WITH([medipack-dir],
  AS_HELP_STRING([--with-medipack-dir=PREFIX],
                 [Prefix where MediPack is installed]),
  [MEDIPACK_DIR=$withval],
  [MEDIPACK_DIR=])

dnl -- Validate combo --------------------------------------------------
if test "x$enable_ad" = "xyes"; then
  if test -z "$CODIPACK_DIR" || test -z "$MEDIPACK_DIR"; then
    AC_MSG_ERROR([--enable-ad requires BOTH --with-codipack-dir and --with-medipack-dir])
  fi
  AC_DEFINE([ISSM_USE_AD], [1], [Define to 1 if building with automatic differentiation])
  ENABLE_PETSC=no
else
  ENABLE_PETSC=yes
fi
AM_CONDITIONAL([USE_AD], [test "x$enable_ad" = "xyes"])

dnl -- Propagate dirs/flags -----------------------------
if test "x$enable_ad" = "xyes"; then
  AM_CPPFLAGS="$AM_CPPFLAGS -I$CODIPACK_DIR/include -I$MEDIPACK_DIR/include -DCODI_ForcedInlines"
  AM_LDFLAGS="$AM_LDFLAGS -L$CODIPACK_DIR/lib -L$MEDIPACK_DIR/lib"
  LIBS="$LIBS -lcodi -lmedi"
fi
AC_SUBST([AM_CPPFLAGS])
AC_SUBST([AM_LDFLAGS])
AC_SUBST([LIBS])

# ---------------------------------------------------------------------
#  Run the standard ISSM option-detection macro collection
# ---------------------------------------------------------------------
ISSM_OPTIONS

# =====================================================================
#  Configure output files
# =====================================================================
AC_CONFIG_FILES([
        Makefile
        src/Makefile
        src/c/Makefile
        src/wrappers/Makefile
        src/wrappers/python/Makefile
        src/wrappers/matlab/Makefile
        src/wrappers/javascript/Makefile
        src/m/Makefile
])

# =====================================================================
#  Finalise
# =====================================================================
AC_OUTPUT
