

PyObject *
cache_info(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int i;
    for (i = 0; i < 32; i++) {

        // Variables to hold the contents of the 4 i386 legacy registers
        int cpu_info[4] = {0};

        __cpuidex(cpu_info, 4, i);


                // See the page 3-191 of the manual.
        int cache_type = cpu_info[0] & 0x1F; 

        if (cache_type == 0) // end of valid cache identifiers
            break;

        char * cache_type_string;
        switch (cache_type) {
            case 1: cache_type_string = "Data Cache"; break;
            case 2: cache_type_string = "Instruction Cache"; break;
            case 3: cache_type_string = "Unified Cache"; break;
            default: cache_type_string = "Unknown Type Cache"; break;
        }

        int cache_level = (cpu_info[0] >>= 5) & 0x7;

        int cache_is_self_initializing = (cpu_info[0] >>= 3) & 0x1; // does not need SW initialization
        int cache_is_fully_associative = (cpu_info[0] >>= 1) & 0x1;

        // See the page 3-192 of the manual.
        // cpu_info[1] contains 3 integers of 10, 10 and 12 bits respectively
        unsigned int cache_sets = cpu_info[2] + 1;
        unsigned int cache_coherency_line_size = (cpu_info[1] & 0xFFF) + 1;
        unsigned int cache_physical_line_partitions = ((cpu_info[1] >>= 12) & 0x3FF) + 1;
        unsigned int cache_ways_of_associativity = ((cpu_info[1] >>= 10) & 0x3FF) + 1;

        // Total cache size is the product
        size_t cache_total_size = cache_ways_of_associativity * cache_physical_line_partitions * cache_coherency_line_size * cache_sets;

        printf(
            "Cache ID %d:\n"
            "- Level: %d\n"
            "- Type: %s\n"
            "- Sets: %d\n"
            "- System Coherency Line Size: %d bytes\n"
            "- Physical Line partitions: %d\n"
            "- Ways of associativity: %d\n"
            "- Total Size: %zu bytes (%zu kb)\n"
            "- Is fully associative: %s\n"
            "- Is Self Initializing: %s\n"
            "\n"
            , i
            , cache_level
            , cache_type_string
            , cache_sets
            , cache_coherency_line_size
            , cache_physical_line_partitions
            , cache_ways_of_associativity
            , cache_total_size, cache_total_size >> 10
            , cache_is_fully_associative ? "true" : "false"
            , cache_is_self_initializing ? "true" : "false"
        );
    }
    Py_RETURN_NONE;
}
