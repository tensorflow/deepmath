/*    
Lists and tuples are separated by either ',' or space.

Some notes on abbreviations:

  - '-xxx' translates into '-xxx=true' (useful for boolean switches)
  - '-no-xxx' translates into '-xxx=false'

Furthermore, 'xxx' (no minus) is treated as a positional parameter and assigned to the argument
of that position. Positions cannot be embedded (i.e. embedded arguments must be specified using
'-name=xxx'). Double minuses are also allowed: '--xxx' is interpreted the same as '-xxx'

Switches '-h' and '-help' are reserved for printing help.

Normally, an argument may only be specified once, but after '--' you are allowed to redefined
arguments (once per argument). You can repeat the use of '--'. The intended purpose is to
allow scripts to set its own default parameters, but have a way of overriding those values.

//=================================================================================================


Subtype relation:

     int
      |
      |
    float bool enum  tuple
       \   |   /      |
        \  |  /       |
         string      list
          \         /
           \       /
           universe

In expressions, list syntax '[1 2]' can be used in place of tuple syntax '(1,2)' if there is
no matching list signature (the list will be implicitly converted). This is convenient for
the shell command line where parenthesis are not preserved. Inside lists or tuples, commas
must be quoted if part of the string value (likewise for characters '()[]' and space).

      
      
==============================
BNF FOR SIGNATURES
==============================

In the BNF, a production rule with suffix 's' means a comma or space separated list (eg. 
"types" can mean "int, int, float" or "int int float" etc.).

A number is an optional sign followed by a either an integer or a floating point number
(using standard C syntax with 'e' or 'E' for exponent).

    num ::= [+-]<int or float>
          
    Examples: -12  +.5  123.456  42e-5

A boolean ('bool') is either TRUE or FALSE, as indicated by any of the following strings,
disregarding case:

    bool ::= '1' | '0' | 'true' | 'false' | 'on' | 'off' | 'yes' | 'no'

    Examples: 1  True  off  YES

A range is lower and an upper bound separated by a colon. If the lower bound is follwed by a '+',
the bound is strict; similarily if the uppe rbound is followed by a '-'. 
          
    range ::= num     ':' num
            | num '+' ':' num
            | num     ':' num '-'
            | num '+' ':' num '-'
            |         ':' num
            |         ':' num '-'
            | num     ':'
            | num '+' ':'
            
    Examples: 0:99  0+:  -0.5+:0.5-            

An atom is either 'int' or 'float', optionally followed by an underscore + range, or an enum list,
or the string 'string'. Non-negative integers or floats are given synonyms 'uint' and 'ufloat'.
    
    atom ::= 'bool'                                     // Boolean; 'bool
           | 'int' | 'uint' | 'int' '[' range ']'       // Integer; 'uint' == 'int[0:]'
           | 'float' | 'ufloat' | 'float' '[' range ']' // Float; 'ufloat' == 'float[0:]'
           | '{' strings '}'                            // Enum
           | 'string'                                   // String

    Examples: int[-10:10]  {slow, fast}  string
           
A type is either atomic or a composite. Composit types are tuples (comma separated list 
of two or more types within parenthesis) or lists (single type within square brackets). The special
type 'any' matches anything. Vertical bar is used for disjunctive ("any one of") types.
    
    type ::= atom
           | '(' types ')'                              // Tuple (must be at least a pair)
           | '[' type ']'                               // List
           | 'any'                                      // Universal type
           | type '|' type                              // Disjunction

    Example: (int, int) | [float | {none}] | string


==============================
BNF FOR VALUES
==============================

    string = Any sequence of character except '()[]{}|,:' or whitespace characters 9-13 or 32 (space).
             These special characters must be quoted by '\' (possibly '\\' on the shell 
             command line). 
             
    list  ::= '[' elems ']'
    tuple ::= '(' elems ')'
    elem  ::= string | list | tuple
          
The universal value produced by parsing this BNF is called "unmatched". Coupled with a signature
that specializes (some) strings to floats and ints or enums and (some) lists to tuples is called
"matching".
    
*/


