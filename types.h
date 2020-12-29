#ifndef TYPES_H
#define TYPES_H

/**
 * Parts materials (not in use)
 */
enum Material {
    /**
     * Fe
     */
    iron=0,
};

/**
 * X-ray source tube types
 */
enum TubeType {
    /**
     * Be 8 mm
     */
    Be_08 = 0,
    /**
     * Be 30 mm
     */
    Be_30,
    /**
     * Be 50 mm
     */
    Be_50,
};

/**
 * Part types (temp)
 */
enum PartType {
    /**
     * A part with a notch
     */
    PT_notch = 0,
    /**
     * A part with a bubble inside it
     */
    PT_bubble,
};

#endif