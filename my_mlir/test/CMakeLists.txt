

# 配置lit套件设置
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
# 
set(TEST_DEPENDS
  NS-opt
)


# 只有这些 target 真存在时才加依赖（兼容 in-tree 构建）
foreach(t FileCheck count not)
  if(TARGET ${t})
    list(APPEND TEST_DEPENDS ${t})
  endif()
endforeach()

add_lit_testsuite(check-ch "Running the lit regression tests..."
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
)

set_target_properties(check-ch PROPERTIES FOLDER "test")

