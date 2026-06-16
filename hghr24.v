// ============================================================
// MODULE : hybrid_hghr_cordic  —  2^Q  Rotation Mode  (24-bit)
// FORMAT : Q4.20  ONE=1048576  SCALE=2^20  range (-8,+8)
// SCHEDULE: R8×4 + R4×2  (6 CORDIC stages)
// INPUT  : Q_fixed [23:0] in Q4.20, Q in [-0.5, 0.5]
// OUTPUT : exp2_Q_fixed [23:0] in Q4.20
// LATENCY: 9 clk  (stage0 + 6 CORDIC + sum_r + prod_reg)
//
// ── MULTIPLY FIX ─────────────────────────────────────────────
// di is declared as full-width signed (24-bit) matching the
// vectoring module style so that di*y and di*x evaluate at 48
// bits before the [S+23:S] slice, avoiding the 4-bit truncation
// bug that plagued earlier attempt.
// ============================================================
`timescale 1ns/1ps
module hybrid_hghr_cordic (
    input  wire        clk, rst_n, valid_in,
    input  wire [23:0] Q_fixed,
    output reg         valid_out,
    output reg  [23:0] exp2_Q_fixed
);
    localparam signed [23:0] ONE = 24'sd1048576; // 1.0 in Q4.20

    // ── Angle LUT: 9 stored constants (Q4.20) ─────────────────
    // Angles are atanh(d/r) / ln(2) — base-2 log scale
    localparam signed [23:0] S1A1 = 24'sd190091;  // atanh(1/8)/ln2
    localparam signed [23:0] S1A2 = 24'sd386382;  // atanh(2/8)/ln2
    localparam signed [23:0] S1A3 = 24'sd596379;  // atanh(3/8)/ln2
    localparam signed [23:0] S1A4 = 24'sd830977;  // atanh(4/8)/ln2
    localparam signed [23:0] S2A1 = 24'sd23639;   // atanh(1/64)/ln2
    localparam signed [23:0] S2A2 = 24'sd47289;   // atanh(2/64)/ln2
    localparam signed [23:0] S2A3 = 24'sd70963;   // atanh(3/64)/ln2
    localparam signed [23:0] S2A4 = 24'sd94672;   // atanh(4/64)/ln2
    localparam signed [23:0] ANC  = 24'sd2955;    // stage-3 anchor

    reg signed [23:0] x0,y0,z0; reg signed [23:0] k0; reg v0;
    reg signed [23:0] x1,y1,z1; reg signed [23:0] k1; reg v1;
    reg signed [23:0] x2,y2,z2; reg signed [23:0] k2; reg v2;
    reg signed [23:0] x3,y3,z3; reg signed [23:0] k3; reg v3;
    reg signed [23:0] x4,y4,z4; reg signed [23:0] k4; reg v4;
    reg signed [23:0] x5,y5,z5; reg signed [23:0] k5; reg v5;
    reg signed [23:0] x6,y6,z6; reg signed [23:0] k6; reg v6;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin x0<=0; y0<=0; z0<=0; k0<=ONE; v0<=0; end
        else begin
            x0 <= ONE;
            y0 <= 24'sd0;
            z0 <= $signed(Q_fixed);
            k0 <= ONE;
            v0 <= valid_in;
        end
    end

    reg [2:0] d1r0;

    // ── Stage 1  S=3  dmax=4 ─────────────────────────────────
    wire [23:0] s1_za = z0[23] ? -z0 : z0;
    wire [2:0] s1_da = (s1_za >= 24'd713678) ? 3'd4 :
                       (s1_za >= 24'd491380) ? 3'd3 :
                       (s1_za >= 24'd288236) ? 3'd2 :
                       (s1_za >= 24'd95045)  ? 3'd1 : 3'd0;
    wire signed [23:0] s1_di  = z0[23] ? (-{{21{1'b0}},s1_da}) : ({{21{1'b0}},s1_da});
    wire signed [23:0] s1_ap  = (s1_da==3'd4)?S1A4:(s1_da==3'd3)?S1A3:
                                (s1_da==3'd2)?S1A2:(s1_da==3'd1)?S1A1:24'sd0;
    wire signed [23:0] s1_ang = s1_di[23] ? -s1_ap : s1_ap;
    wire signed [47:0] s1_py  = $signed(s1_di)*$signed({{24{y0[23]}},y0});
    wire signed [47:0] s1_px  = $signed(s1_di)*$signed({{24{x0[23]}},x0});
    wire signed [23:0] s1_xn  = x0 + s1_py[26:3];
    wire signed [23:0] s1_yn  = y0 + s1_px[26:3];
    wire signed [23:0] s1_zn  = z0 - s1_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) d1r0 <= 0; else d1r0 <= s1_da;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x1<=0; y1<=0; z1<=0; k1<=ONE; v1<=0; end
        else begin x1<=s1_xn; y1<=s1_yn; z1<=s1_zn; k1<=k0; v1<=v0; end

    // ── Stage 2  S=6  dmax=4 ─────────────────────────────────
    wire [23:0] s2_za = z1[23] ? -z1 : z1;
    wire [2:0] s2_da = (s2_za >= 24'd82817) ? 3'd4 :
                       (s2_za >= 24'd59126) ? 3'd3 :
                       (s2_za >= 24'd35464) ? 3'd2 :
                       (s2_za >= 24'd11819) ? 3'd1 : 3'd0;
    wire signed [23:0] s2_di  = z1[23] ? (-{{21{1'b0}},s2_da}) : ({{21{1'b0}},s2_da});
    wire signed [23:0] s2_ap  = (s2_da==3'd4)?S2A4:(s2_da==3'd3)?S2A3:
                                (s2_da==3'd2)?S2A2:(s2_da==3'd1)?S2A1:24'sd0;
    wire signed [23:0] s2_ang = s2_di[23] ? -s2_ap : s2_ap;
    wire signed [47:0] s2_py  = $signed(s2_di)*$signed({{24{y1[23]}},y1});
    wire signed [47:0] s2_px  = $signed(s2_di)*$signed({{24{x1[23]}},x1});
    wire signed [23:0] s2_xn  = x1 + s2_py[29:6];
    wire signed [23:0] s2_yn  = y1 + s2_px[29:6];
    wire signed [23:0] s2_zn  = z1 - s2_ang;

    // Kinv LUT: 25 entries scaled from Q4.28 to Q4.20 (÷256)
    wire signed [23:0] s2_kn =
        (d1r0==3'd4 && s2_da==3'd4) ? 24'sd1213163 :
        (d1r0==3'd4 && s2_da==3'd3) ? 24'sd1212123 :
        (d1r0==3'd4 && s2_da==3'd2) ? 24'sd1211382 :
        (d1r0==3'd4 && s2_da==3'd1) ? 24'sd1210939 :
        (d1r0==3'd4 && s2_da==3'd0) ? 24'sd1210791 :
        (d1r0==3'd3 && s2_da==3'd4) ? 24'sd1133335 :
        (d1r0==3'd3 && s2_da==3'd3) ? 24'sd1132364 :
        (d1r0==3'd3 && s2_da==3'd2) ? 24'sd1131672 :
        (d1r0==3'd3 && s2_da==3'd1) ? 24'sd1131257 :
        (d1r0==3'd3 && s2_da==3'd0) ? 24'sd1131119 :
        (d1r0==3'd2 && s2_da==3'd4) ? 24'sd1085086 :
        (d1r0==3'd2 && s2_da==3'd3) ? 24'sd1084156 :
        (d1r0==3'd2 && s2_da==3'd2) ? 24'sd1083493 :
        (d1r0==3'd2 && s2_da==3'd1) ? 24'sd1083096 :
        (d1r0==3'd2 && s2_da==3'd0) ? 24'sd1082964 :
        (d1r0==3'd1 && s2_da==3'd4) ? 24'sd1058935 :
        (d1r0==3'd1 && s2_da==3'd3) ? 24'sd1058028 :
        (d1r0==3'd1 && s2_da==3'd2) ? 24'sd1057381 :
        (d1r0==3'd1 && s2_da==3'd1) ? 24'sd1056994 :
        (d1r0==3'd1 && s2_da==3'd0) ? 24'sd1056865 :
        (d1r0==3'd0 && s2_da==3'd4) ? 24'sd1050630 :
        (d1r0==3'd0 && s2_da==3'd3) ? 24'sd1049729 :
        (d1r0==3'd0 && s2_da==3'd2) ? 24'sd1049088 :
        (d1r0==3'd0 && s2_da==3'd1) ? 24'sd1048704 : 24'sd1048576;

    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x2<=0; y2<=0; z2<=0; k2<=ONE; v2<=0; end
        else begin x2<=s2_xn; y2<=s2_yn; z2<=s2_zn; k2<=s2_kn; v2<=v1; end

    // ── Stage 3  S=9  dmax=4  (ap via ANC shift-add) ─────────
    wire [23:0] s3_za = z2[23] ? -z2 : z2;
    wire [2:0] s3_da = (s3_za >= 24'd10341) ? 3'd4 :
                       (s3_za >= 24'd7386)  ? 3'd3 :
                       (s3_za >= 24'd4431)  ? 3'd2 :
                       (s3_za >= 24'd1477)  ? 3'd1 : 3'd0;
    wire signed [23:0] s3_di  = z2[23] ? (-{{21{1'b0}},s3_da}) : ({{21{1'b0}},s3_da});
    wire signed [23:0] s3_ap  = (s3_da==3'd4)?(ANC<<<2):
                                (s3_da==3'd3)?(ANC+(ANC<<<1)):
                                (s3_da==3'd2)?(ANC<<<1):
                                (s3_da==3'd1)? ANC:24'sd0;
    wire signed [23:0] s3_ang = s3_di[23] ? -s3_ap : s3_ap;
    wire signed [47:0] s3_py  = $signed(s3_di)*$signed({{24{y2[23]}},y2});
    wire signed [47:0] s3_px  = $signed(s3_di)*$signed({{24{x2[23]}},x2});
    wire signed [23:0] s3_xn  = x2 + s3_py[32:9];
    wire signed [23:0] s3_yn  = y2 + s3_px[32:9];
    wire signed [23:0] s3_zn  = z2 - s3_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x3<=0; y3<=0; z3<=0; k3<=ONE; v3<=0; end
        else begin x3<=s3_xn; y3<=s3_yn; z3<=s3_zn; k3<=k2; v3<=v2; end

    // ── Stage 4  S=12  dmax=4  (base=ANC>>3=369) ─────────────
    wire signed [23:0] s4_b   = ANC >>> 3;
    wire [23:0] s4_za = z3[23] ? -z3 : z3;
    wire [2:0] s4_da = (s4_za >= 24'd1292) ? 3'd4 :
                       (s4_za >= 24'd923)  ? 3'd3 :
                       (s4_za >= 24'd553)  ? 3'd2 :
                       (s4_za >= 24'd184)  ? 3'd1 : 3'd0;
    wire signed [23:0] s4_di  = z3[23] ? (-{{21{1'b0}},s4_da}) : ({{21{1'b0}},s4_da});
    wire signed [23:0] s4_ap  = (s4_da==3'd4)?(s4_b<<<2):
                                (s4_da==3'd3)?(s4_b+(s4_b<<<1)):
                                (s4_da==3'd2)?(s4_b<<<1):
                                (s4_da==3'd1)? s4_b:24'sd0;
    wire signed [23:0] s4_ang = s4_di[23] ? -s4_ap : s4_ap;
    wire signed [47:0] s4_py  = $signed(s4_di)*$signed({{24{y3[23]}},y3});
    wire signed [47:0] s4_px  = $signed(s4_di)*$signed({{24{x3[23]}},x3});
    wire signed [23:0] s4_xn  = x3 + s4_py[35:12];
    wire signed [23:0] s4_yn  = y3 + s4_px[35:12];
    wire signed [23:0] s4_zn  = z3 - s4_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x4<=0; y4<=0; z4<=0; k4<=ONE; v4<=0; end
        else begin x4<=s4_xn; y4<=s4_yn; z4<=s4_zn; k4<=k3; v4<=v3; end

    // ── Stage 5  S=14  dmax=2  (base=ANC>>5=92) ─────────────
    wire signed [23:0] s5_b   = ANC >>> 5;
    wire [23:0] s5_za = z4[23] ? -z4 : z4;
    wire [2:0] s5_da = (s5_za >= 24'd138) ? 3'd2 :
                       (s5_za >= 24'd46)  ? 3'd1 : 3'd0;
    wire signed [23:0] s5_di  = z4[23] ? (-{{21{1'b0}},s5_da}) : ({{21{1'b0}},s5_da});
    wire signed [23:0] s5_ap  = (s5_da==3'd2)?(s5_b<<<1):
                                (s5_da==3'd1)? s5_b:24'sd0;
    wire signed [23:0] s5_ang = s5_di[23] ? -s5_ap : s5_ap;
    wire signed [47:0] s5_py  = $signed(s5_di)*$signed({{24{y4[23]}},y4});
    wire signed [47:0] s5_px  = $signed(s5_di)*$signed({{24{x4[23]}},x4});
    wire signed [23:0] s5_xn  = x4 + s5_py[37:14];
    wire signed [23:0] s5_yn  = y4 + s5_px[37:14];
    wire signed [23:0] s5_zn  = z4 - s5_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x5<=0; y5<=0; z5<=0; k5<=ONE; v5<=0; end
        else begin x5<=s5_xn; y5<=s5_yn; z5<=s5_zn; k5<=k4; v5<=v4; end

    // ── Stage 6  S=16  dmax=2  (base=ANC>>7=23) ─────────────
    wire signed [23:0] s6_b   = ANC >>> 7;
    wire [23:0] s6_za = z5[23] ? -z5 : z5;
    wire [2:0] s6_da = (s6_za >= 24'd34) ? 3'd2 :
                       (s6_za >= 24'd11) ? 3'd1 : 3'd0;
    wire signed [23:0] s6_di  = z5[23] ? (-{{21{1'b0}},s6_da}) : ({{21{1'b0}},s6_da});
    wire signed [23:0] s6_ap  = (s6_da==3'd2)?(s6_b<<<1):
                                (s6_da==3'd1)? s6_b:24'sd0;
    wire signed [23:0] s6_ang = s6_di[23] ? -s6_ap : s6_ap;
    wire signed [47:0] s6_py  = $signed(s6_di)*$signed({{24{y5[23]}},y5});
    wire signed [47:0] s6_px  = $signed(s6_di)*$signed({{24{x5[23]}},x5});
    wire signed [23:0] s6_xn  = x5 + s6_py[39:16];
    wire signed [23:0] s6_yn  = y5 + s6_px[39:16];
    wire signed [23:0] s6_zn  = z5 - s6_ang;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin x6<=0; y6<=0; z6<=0; k6<=ONE; v6<=0; end
        else begin x6<=s6_xn; y6<=s6_yn; z6<=s6_zn; k6<=k5; v6<=v5; end

    // ── Output: 2^Q = (x6 + y6) * kinv >> 20 ────────────────
    reg signed [23:0] sr; reg signed [23:0] kr; reg vs;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin sr<=0; kr<=ONE; vs<=0; end
        else begin sr<=x6+y6; kr<=k6; vs<=v6; end
    wire signed [47:0] prod = $signed(sr) * $signed(kr);
    always @(posedge clk or negedge rst_n)
        if (!rst_n) begin valid_out<=0; exp2_Q_fixed<=0; end
        else begin valid_out<=vs; exp2_Q_fixed<=prod[43:20]; end

endmodule
