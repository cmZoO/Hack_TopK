add_library(SCI-Millionaire INTERFACE)
target_link_libraries(SCI-Millionaire
	INTERFACE SCI-OTPrimitive
)
target_compile_definitions(SCI-Millionaire INTERFACE USE_CHEETAH=${USE_CHEETAH})
