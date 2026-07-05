#ifndef CONVERSION_NORTHSTARTOLINALG_NORTHSTARTOLINALG_H
#define CONVERSION_NORTHSTARTOLINALG_NORTHSTARTOLINALG_H
#include <memory>

#include "mlir/Pass/Pass.h"
namespace mlir {
class TypeConverter;
}

namespace mlir::north_star {

void initNorthStarToLinalgTypeConvert(TypeConverter &typeConverter);

void populateNorthStarToLinalgPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

#define GEN_PASS_DECL_CONVERTNORTHSTARTOLINALGPASS
#include "Conversion/Passes.h.inc"

}  // namespace mlir::north_star
#endif  // CONVERSION_NORTHSTARTOLINALG_NORTHSTARTOLINALG_H
