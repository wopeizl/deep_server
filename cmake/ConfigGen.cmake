
################################################################################################
# Helper function to get all list items that begin with given prefix
# Usage:
#   deep_get_items_with_prefix(<prefix> <list_variable> <output_variable>)
function(deep_get_items_with_prefix prefix list_variable output_variable)
  set(__result "")
  foreach(__e ${${list_variable}})
    if(__e MATCHES "^${prefix}.*")
      list(APPEND __result ${__e})
    endif()
  endforeach()
  set(${output_variable} ${__result} PARENT_SCOPE)
endfunction()

################################################################################################
# Function for generation Deep build- and install- tree export config files
# Usage:
#  deep_generate_export_configs()
function(deep_generate_export_configs)
  set(install_cmake_suffix "share/Deep")

  if(NOT HAVE_CUDA)
    set(HAVE_CUDA FALSE)
  endif()

  set(GFLAGS_IMPORTED OFF)
  foreach(_lib ${GFLAGS_LIBRARIES})
    if(TARGET ${_lib})
      set(GFLAGS_IMPORTED ON)
    endif()
  endforeach()

  set(GLOG_IMPORTED OFF)
  foreach(_lib ${GLOG_LIBRARIES})
    if(TARGET ${_lib})
      set(GLOG_IMPORTED ON)
    endif()
  endforeach()

  set(HDF5_IMPORTED OFF)
  foreach(_lib ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES})
    if(TARGET ${_lib})
      set(HDF5_IMPORTED ON)
    endif()
  endforeach()

  set(LMDB_IMPORTED OFF)
  if(USE_LMDB)
    foreach(_lib ${LMDB_LIBRARIES})
      if(TARGET ${_lib})
        set(LMDB_IMPORTED ON)
      endif()
    endforeach()
  endif()
  set(LEVELDB_IMPORTED OFF)
  set(SNAPPY_IMPORTED OFF)
  if(USE_LEVELDB)
    foreach(_lib ${LevelDB_LIBRARIES})
      if(TARGET ${_lib})
        set(LEVELDB_IMPORTED ON)
      endif()
    endforeach()
    foreach(_lib ${Snappy_LIBRARIES})
      if(TARGET ${_lib})
        set(SNAPPY_IMPORTED ON)
      endif()
    endforeach()
  endif()

  if(NOT HAVE_CUDNN)
    set(HAVE_CUDNN FALSE)
  endif()

  # ---[ Configure build-tree DeepConfig.cmake file ]---

  configure_file("cmake/Templates/DeepConfig.cmake.in" "${PROJECT_BINARY_DIR}/DeepConfig.cmake" @ONLY)

  # Add targets to the build-tree export set
  export(TARGETS deep_server proto FILE "${PROJECT_BINARY_DIR}/DeepTargets.cmake")
  export(PACKAGE deep_server)

  # ---[ Configure install-tree DeepConfig.cmake file ]---

  configure_file("cmake/Templates/DeepConfig.cmake.in" "${PROJECT_BINARY_DIR}/cmake/DeepConfig.cmake" @ONLY)

  # Install the DeepConfig.cmake and export set to use with install-tree
  install(FILES "${PROJECT_BINARY_DIR}/cmake/DeepConfig.cmake" DESTINATION ${install_cmake_suffix})
  install(EXPORT DeepTargets DESTINATION ${install_cmake_suffix})

  # ---[ Configure and install version file ]---

  # TODO: Lines below are commented because Deep doesn't declare its version in headers.
  # When the declarations are added, modify `deep_extract_deep_version()` macro and uncomment

  # configure_file(cmake/Templates/DeepConfigVersion.cmake.in "${PROJECT_BINARY_DIR}/DeepConfigVersion.cmake" @ONLY)
  # install(FILES "${PROJECT_BINARY_DIR}/DeepConfigVersion.cmake" DESTINATION ${install_cmake_suffix})
endfunction()


