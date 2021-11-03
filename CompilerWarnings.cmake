function(set_project_warnings project_name)
	option(WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)
	
	set(GCC_WARNINGS
		-Wall
		-Wextra
		-Wshadow
		-Wnon-virtual-dtor
		-Wold-style-cast
		-Wcast-align
		-Wunused
		-Woverloaded-virtual
		-Wpedantic
		-Wconversion
		-Wsign-conversion
		-Wnull-dereference
		-Wdouble-promotion
		-Wformat=2
		-Wimplicit-fallthrough
		-Wmisleading-indentation
		-Wduplicated-cond
		-Wduplicated-branches
		-Wlogical-op
		-Wuseless-cast
	)
	
	if(WARNINGS_AS_ERRORS)
		set(GCC_WARNINGS ${GCC_WARNINGS} -Werror)
	endif()
	
	if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(PROJECT_WARNINGS ${GCC_WARNINGS})
	else()
		message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
	endif()	
	
	target_compile_options(${project_name} INTERFACE ${PROJECT_WARNINGS})
	#message(AUTHOR_WARNING "Tried to set warnings")
endfunction()