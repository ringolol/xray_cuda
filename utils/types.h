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
     * empty tube type
     */
    TT_none = 0,
    /**
     * Be 8 mm
     */
    TT_Be_08,
    /**
     * Be 30 mm
     */
    TT_Be_30,
    /**
     * Be 50 mm
     */
    TT_Be_50,
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

#endif  // TYPES_H