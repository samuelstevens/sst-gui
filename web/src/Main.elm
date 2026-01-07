module Main exposing (..)

import Browser
import Dict exposing (Dict)
import File exposing (File)
import Html exposing (..)
import Html.Attributes exposing (..)
import Html.Events exposing (..)
import Http
import Json.Decode as D
import Json.Encode as E
import Set exposing (Set)



-- MAIN


main : Program () Model Msg
main =
    Browser.element
        { init = init
        , update = update
        , subscriptions = \_ -> Sub.none
        , view = view
        }



-- MODEL


type alias Model =
    { page : Page
    , error : Maybe String

    -- Project setup form
    , csvFile : Maybe File
    , filterQuery : String
    , groupBy : String
    , imgPath : String
    , primaryKey : String
    , rootDpath : String
    , sam2Model : String
    , device : String

    -- Project state
    , projectId : Maybe String
    , columns : List String
    , groupCount : Int
    , rowCount : Int

    -- Groups
    , groups : List GroupSummary
    , selectedGroupKey : Maybe String

    -- Sampling
    , nRefFrames : String
    , seed : String

    -- Frames
    , frames : List FrameSummary
    , currentFrameIndex : Int

    -- Masks
    , masks : List MaskMeta
    , selectedMasks : Dict Int Int -- mask_id -> label
    , maskScale : Float
    , loadingMasks : Bool
    }


type Page
    = SetupPage
    | GroupsPage
    | FramesPage
    | MasksPage


type alias GroupSummary =
    { groupKey : String
    , groupDisplay : Dict String String
    , count : Int
    }


type alias FrameSummary =
    { pk : String
    , imgPath : String
    , masksCached : Bool
    }


type alias MaskMeta =
    { maskId : Int
    , score : Maybe Float
    , area : Maybe Int
    , url : String
    }


init : () -> ( Model, Cmd Msg )
init _ =
    ( { page = SetupPage
      , error = Nothing
      , csvFile = Nothing
      , filterQuery = "SELECT * FROM master_df"
      , groupBy = "Dataset"
      , imgPath = "CONCAT('/local/scratch/datasets/jiggins/butterflies', RIGHT(filepath, LEN(filepath) - 6))"
      , primaryKey = "Image_name"
      , rootDpath = "/local/scratch/stevens.994/datasets/cambridge-segmented"
      , sam2Model = "facebook/sam2.1-hiera-tiny"
      , device = "cuda"
      , projectId = Nothing
      , columns = []
      , groupCount = 0
      , rowCount = 0
      , groups = []
      , selectedGroupKey = Nothing
      , nRefFrames = "5"
      , seed = "0"
      , frames = []
      , currentFrameIndex = 0
      , masks = []
      , selectedMasks = Dict.empty
      , maskScale = 0.25
      , loadingMasks = False
      }
    , Cmd.none
    )



-- MSG


type Msg
    = -- Setup form
      CsvSelected File
    | SetFilterQuery String
    | SetGroupBy String
    | SetImgPath String
    | SetPrimaryKey String
    | SetRootDpath String
    | SetSam2Model String
    | SetDevice String
    | SubmitProject
    | GotProjectCreated (Result Http.Error ProjectCreatedResponse)
      -- Groups
    | GotGroups (Result Http.Error GroupsResponse)
    | SelectGroup String
    | SetNRefFrames String
    | SetSeed String
    | SampleFrames
    | GotSampledFrames (Result Http.Error SampledFramesResponse)
      -- Frames
    | GoToFrame Int
    | ComputeMasks
    | GotMasks (Result Http.Error MasksResponse)
      -- Mask selection
    | ToggleMask Int
    | SetMaskLabel Int String
    | SaveSelection
    | GotSelectionSaved (Result Http.Error SelectionSavedResponse)
      -- Navigation
    | GoToPage Page
    | DismissError


type alias ProjectCreatedResponse =
    { projectId : String
    , columns : List String
    , groupCount : Int
    , rowCount : Int
    }


type alias GroupsResponse =
    { groups : List GroupSummary
    , total : Int
    }


type alias SampledFramesResponse =
    { frames : List FrameSummary
    }


type alias MasksResponse =
    { scale : Float
    , masks : List MaskMeta
    }


type alias SelectionSavedResponse =
    { savedFpath : String
    }



-- UPDATE


update : Msg -> Model -> ( Model, Cmd Msg )
update msg model =
    case msg of
        CsvSelected file ->
            ( { model | csvFile = Just file }, Cmd.none )

        SetFilterQuery val ->
            ( { model | filterQuery = val }, Cmd.none )

        SetGroupBy val ->
            ( { model | groupBy = val }, Cmd.none )

        SetImgPath val ->
            ( { model | imgPath = val }, Cmd.none )

        SetPrimaryKey val ->
            ( { model | primaryKey = val }, Cmd.none )

        SetRootDpath val ->
            ( { model | rootDpath = val }, Cmd.none )

        SetSam2Model val ->
            ( { model | sam2Model = val }, Cmd.none )

        SetDevice val ->
            ( { model | device = val }, Cmd.none )

        SubmitProject ->
            case model.csvFile of
                Just file ->
                    ( model, createProject model file )

                Nothing ->
                    ( { model | error = Just "Please select a CSV file" }, Cmd.none )

        GotProjectCreated result ->
            case result of
                Ok resp ->
                    ( { model
                        | projectId = Just resp.projectId
                        , columns = resp.columns
                        , groupCount = resp.groupCount
                        , rowCount = resp.rowCount
                        , page = GroupsPage
                        , error = Nothing
                      }
                    , fetchGroups resp.projectId
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GotGroups result ->
            case result of
                Ok resp ->
                    ( { model | groups = resp.groups, error = Nothing }, Cmd.none )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        SelectGroup groupKey ->
            ( { model | selectedGroupKey = Just groupKey }, Cmd.none )

        SetNRefFrames val ->
            ( { model | nRefFrames = val }, Cmd.none )

        SetSeed val ->
            ( { model | seed = val }, Cmd.none )

        SampleFrames ->
            case ( model.projectId, model.selectedGroupKey ) of
                ( Just pid, Just gk ) ->
                    ( model, sampleFrames pid gk model.nRefFrames model.seed )

                _ ->
                    ( { model | error = Just "Select a group first" }, Cmd.none )

        GotSampledFrames result ->
            case result of
                Ok resp ->
                    ( { model
                        | frames = resp.frames
                        , currentFrameIndex = 0
                        , page = FramesPage
                        , error = Nothing
                        , masks = []
                        , selectedMasks = Dict.empty
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GoToFrame idx ->
            if idx >= 0 && idx < List.length model.frames then
                ( { model
                    | currentFrameIndex = idx
                    , masks = []
                    , selectedMasks = Dict.empty
                  }
                , Cmd.none
                )

            else
                ( model, Cmd.none )

        ComputeMasks ->
            case model.projectId of
                Just pid ->
                    case List.head (List.drop model.currentFrameIndex model.frames) of
                        Just frame ->
                            ( { model | loadingMasks = True }, computeMasks pid frame.pk model.maskScale )

                        Nothing ->
                            ( model, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        GotMasks result ->
            case result of
                Ok resp ->
                    ( { model
                        | masks = resp.masks
                        , loadingMasks = False
                        , page = MasksPage
                        , error = Nothing
                      }
                    , Cmd.none
                    )

                Err err ->
                    ( { model | error = Just (httpErrorToString err), loadingMasks = False }, Cmd.none )

        ToggleMask maskId ->
            let
                newSelected =
                    if Dict.member maskId model.selectedMasks then
                        Dict.remove maskId model.selectedMasks

                    else
                        Dict.insert maskId 1 model.selectedMasks
            in
            ( { model | selectedMasks = newSelected }, Cmd.none )

        SetMaskLabel maskId labelStr ->
            case String.toInt labelStr of
                Just label ->
                    ( { model | selectedMasks = Dict.insert maskId label model.selectedMasks }, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        SaveSelection ->
            case model.projectId of
                Just pid ->
                    case List.head (List.drop model.currentFrameIndex model.frames) of
                        Just frame ->
                            ( model, saveSelection pid frame.pk model.selectedMasks )

                        Nothing ->
                            ( model, Cmd.none )

                Nothing ->
                    ( model, Cmd.none )

        GotSelectionSaved result ->
            case result of
                Ok _ ->
                    let
                        nextIndex =
                            model.currentFrameIndex + 1
                    in
                    if nextIndex < List.length model.frames then
                        ( { model
                            | currentFrameIndex = nextIndex
                            , masks = []
                            , selectedMasks = Dict.empty
                            , page = FramesPage
                            , error = Nothing
                          }
                        , Cmd.none
                        )

                    else
                        ( { model
                            | error = Nothing
                            , page = GroupsPage
                            , masks = []
                            , selectedMasks = Dict.empty
                          }
                        , Cmd.none
                        )

                Err err ->
                    ( { model | error = Just (httpErrorToString err) }, Cmd.none )

        GoToPage page ->
            ( { model | page = page }, Cmd.none )

        DismissError ->
            ( { model | error = Nothing }, Cmd.none )



-- HTTP


createProject : Model -> File -> Cmd Msg
createProject model file =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects"
        , body =
            Http.multipartBody
                [ Http.filePart "csv" file
                , Http.stringPart "filter_query" model.filterQuery
                , Http.stringPart "group_by" model.groupBy
                , Http.stringPart "img_path" model.imgPath
                , Http.stringPart "primary_key" model.primaryKey
                , Http.stringPart "root_dpath" model.rootDpath
                , Http.stringPart "sam2_model" model.sam2Model
                , Http.stringPart "device" model.device
                ]
        , expect = Http.expectJson GotProjectCreated projectCreatedDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


fetchGroups : String -> Cmd Msg
fetchGroups projectId =
    Http.get
        { url = "/api/projects/" ++ projectId ++ "/groups?limit=1000"
        , expect = Http.expectJson GotGroups groupsDecoder
        }


sampleFrames : String -> String -> String -> String -> Cmd Msg
sampleFrames projectId groupKey nRefFrames seed =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/groups/" ++ groupKey ++ "/sample"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "n_ref_frames", E.int (Maybe.withDefault 5 (String.toInt nRefFrames)) )
                    , ( "seed", E.int (Maybe.withDefault 0 (String.toInt seed)) )
                    ]
                )
        , expect = Http.expectJson GotSampledFrames sampledFramesDecoder
        , timeout = Nothing
        , tracker = Nothing
        }


computeMasks : String -> String -> Float -> Cmd Msg
computeMasks projectId pk scale =
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/frames/" ++ pk ++ "/masks"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "scale", E.float scale )
                    ]
                )
        , expect = Http.expectJson GotMasks masksDecoder
        , timeout = Just 120000
        , tracker = Nothing
        }


saveSelection : String -> String -> Dict Int Int -> Cmd Msg
saveSelection projectId pk selectedMasks =
    let
        items =
            Dict.toList selectedMasks

        maskIds =
            List.map Tuple.first items

        labels =
            List.map Tuple.second items
    in
    Http.request
        { method = "POST"
        , headers = []
        , url = "/api/projects/" ++ projectId ++ "/frames/" ++ pk ++ "/selection"
        , body =
            Http.jsonBody
                (E.object
                    [ ( "mask_ids", E.list E.int maskIds )
                    , ( "labels", E.list E.int labels )
                    ]
                )
        , expect = Http.expectJson GotSelectionSaved selectionSavedDecoder
        , timeout = Nothing
        , tracker = Nothing
        }



-- DECODERS


projectCreatedDecoder : D.Decoder ProjectCreatedResponse
projectCreatedDecoder =
    D.map4 ProjectCreatedResponse
        (D.field "project_id" D.string)
        (D.field "columns" (D.list D.string))
        (D.field "group_count" D.int)
        (D.field "row_count" D.int)


groupsDecoder : D.Decoder GroupsResponse
groupsDecoder =
    D.map2 GroupsResponse
        (D.field "groups" (D.list groupSummaryDecoder))
        (D.field "total" D.int)


groupSummaryDecoder : D.Decoder GroupSummary
groupSummaryDecoder =
    D.map3 GroupSummary
        (D.field "group_key" D.string)
        (D.field "group_display" (D.dict D.string))
        (D.field "count" D.int)


sampledFramesDecoder : D.Decoder SampledFramesResponse
sampledFramesDecoder =
    D.map SampledFramesResponse
        (D.field "frames" (D.list frameSummaryDecoder))


frameSummaryDecoder : D.Decoder FrameSummary
frameSummaryDecoder =
    D.map3 FrameSummary
        (D.field "pk" D.string)
        (D.field "img_path" D.string)
        (D.oneOf [ D.field "masks_cached" D.bool, D.succeed False ])


masksDecoder : D.Decoder MasksResponse
masksDecoder =
    D.map2 MasksResponse
        (D.field "scale" D.float)
        (D.field "masks" (D.list maskMetaDecoder))


maskMetaDecoder : D.Decoder MaskMeta
maskMetaDecoder =
    D.map4 MaskMeta
        (D.field "mask_id" D.int)
        (D.maybe (D.field "score" D.float))
        (D.maybe (D.field "area" D.int))
        (D.field "url" D.string)


selectionSavedDecoder : D.Decoder SelectionSavedResponse
selectionSavedDecoder =
    D.map SelectionSavedResponse
        (D.field "saved_fpath" D.string)


httpErrorToString : Http.Error -> String
httpErrorToString err =
    case err of
        Http.BadUrl url ->
            "Bad URL: " ++ url

        Http.Timeout ->
            "Request timed out"

        Http.NetworkError ->
            "Network error"

        Http.BadStatus status ->
            "Bad status: " ++ String.fromInt status

        Http.BadBody body ->
            "Bad body: " ++ body



-- VIEW


view : Model -> Html Msg
view model =
    div [ class "max-w-6xl mx-auto p-4" ]
        [ viewError model.error
        , case model.page of
            SetupPage ->
                viewSetupPage model

            GroupsPage ->
                viewGroupsPage model

            FramesPage ->
                viewFramesPage model

            MasksPage ->
                viewMasksPage model
        ]


viewError : Maybe String -> Html Msg
viewError maybeError =
    case maybeError of
        Just err ->
            div [ class "bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 flex justify-between items-center" ]
                [ span [] [ text err ]
                , button [ onClick DismissError, class "text-red-700 font-bold" ] [ text "x" ]
                ]

        Nothing ->
            text ""


viewSetupPage : Model -> Html Msg
viewSetupPage model =
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Project Setup" ]
        , div [ class "space-y-4" ]
            [ div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "CSV File" ]
                , input
                    [ type_ "file"
                    , accept ".csv"
                    , on "change" (D.map CsvSelected fileDecoder)
                    , class "block w-full text-sm border rounded p-2"
                    ]
                    []
                ]
            , viewInput "Filter Query (SQL)" model.filterQuery SetFilterQuery
            , viewInput "Group By (comma-separated columns)" model.groupBy SetGroupBy
            , viewInput "Image Path (column or expression)" model.imgPath SetImgPath
            , viewInput "Primary Key (column)" model.primaryKey SetPrimaryKey
            , viewInput "Root Directory" model.rootDpath SetRootDpath
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "SAM2 Model" ]
                , select
                    [ onInput SetSam2Model
                    , value model.sam2Model
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "facebook/sam2.1-hiera-tiny" ] [ text "sam2.1-hiera-tiny" ]
                    , option [ value "facebook/sam2.1-hiera-small" ] [ text "sam2.1-hiera-small" ]
                    , option [ value "facebook/sam2.1-hiera-base-plus" ] [ text "sam2.1-hiera-base-plus" ]
                    , option [ value "facebook/sam2.1-hiera-large" ] [ text "sam2.1-hiera-large" ]
                    ]
                ]
            , div []
                [ label [ class "block text-sm font-medium mb-1" ] [ text "Device" ]
                , select
                    [ onInput SetDevice
                    , value model.device
                    , class "block w-full border rounded p-2"
                    ]
                    [ option [ value "cuda" ] [ text "cuda" ]
                    , option [ value "cpu" ] [ text "cpu" ]
                    ]
                ]
            , button
                [ onClick SubmitProject
                , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
                ]
                [ text "Create Project" ]
            ]
        ]


viewInput : String -> String -> (String -> Msg) -> Html Msg
viewInput labelText val toMsg =
    div []
        [ label [ class "block text-sm font-medium mb-1" ] [ text labelText ]
        , input
            [ type_ "text"
            , value val
            , onInput toMsg
            , class "block w-full border rounded p-2"
            ]
            []
        ]


viewGroupsPage : Model -> Html Msg
viewGroupsPage model =
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Select Group" ]
        , div [ class "mb-4 text-sm text-gray-600" ]
            [ text ("Project: " ++ Maybe.withDefault "" model.projectId)
            , text (" | " ++ String.fromInt model.rowCount ++ " rows, " ++ String.fromInt model.groupCount ++ " groups")
            ]
        , div [ class "grid grid-cols-2 gap-4 mb-4" ]
            [ viewInput "Number of reference frames" model.nRefFrames SetNRefFrames
            , viewInput "Random seed" model.seed SetSeed
            ]
        , div [ class "border rounded max-h-96 overflow-y-auto" ]
            (List.map (viewGroupRow model.selectedGroupKey) model.groups)
        , div [ class "mt-4" ]
            [ button
                [ onClick SampleFrames
                , disabled (model.selectedGroupKey == Nothing)
                , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300"
                ]
                [ text "Sample Frames" ]
            ]
        ]


viewGroupRow : Maybe String -> GroupSummary -> Html Msg
viewGroupRow selectedKey group =
    let
        isSelected =
            selectedKey == Just group.groupKey

        displayText =
            group.groupDisplay
                |> Dict.toList
                |> List.map (\( k, v ) -> k ++ "=" ++ v)
                |> String.join ", "
    in
    div
        [ onClick (SelectGroup group.groupKey)
        , class
            ("p-3 border-b cursor-pointer hover:bg-gray-100 "
                ++ (if isSelected then
                        "bg-blue-100"

                    else
                        ""
                   )
            )
        ]
        [ span [ class "font-medium" ] [ text displayText ]
        , span [ class "text-gray-500 ml-2" ] [ text ("(" ++ String.fromInt group.count ++ " frames)") ]
        ]


viewFramesPage : Model -> Html Msg
viewFramesPage model =
    let
        currentFrame =
            List.head (List.drop model.currentFrameIndex model.frames)
    in
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Frames" ]
        , div [ class "mb-4 text-sm text-gray-600" ]
            [ text ("Frame " ++ String.fromInt (model.currentFrameIndex + 1) ++ " of " ++ String.fromInt (List.length model.frames)) ]
        , case currentFrame of
            Just frame ->
                div []
                    [ div [ class "mb-4" ]
                        [ case model.projectId of
                            Just pid ->
                                img
                                    [ src ("/api/projects/" ++ pid ++ "/frames/" ++ frame.pk ++ "/image?scale=" ++ String.fromFloat model.maskScale)
                                    , class "max-w-full border rounded"
                                    ]
                                    []

                            Nothing ->
                                text ""
                        ]
                    , div [ class "text-sm text-gray-600 mb-4" ] [ text ("PK: " ++ frame.pk) ]
                    , div [ class "flex gap-2" ]
                        [ button
                            [ onClick (GoToFrame (model.currentFrameIndex - 1))
                            , disabled (model.currentFrameIndex == 0)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300 disabled:bg-gray-100"
                            ]
                            [ text "Previous" ]
                        , button
                            [ onClick ComputeMasks
                            , disabled model.loadingMasks
                            , class "bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-300"
                            ]
                            [ text
                                (if model.loadingMasks then
                                    "Computing..."

                                 else
                                    "Compute Masks"
                                )
                            ]
                        , button
                            [ onClick (GoToFrame (model.currentFrameIndex + 1))
                            , disabled (model.currentFrameIndex >= List.length model.frames - 1)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300 disabled:bg-gray-100"
                            ]
                            [ text "Next" ]
                        ]
                    ]

            Nothing ->
                text "No frames"
        , div [ class "mt-4" ]
            [ button
                [ onClick (GoToPage GroupsPage)
                , class "text-blue-500 hover:underline"
                ]
                [ text "Back to groups" ]
            ]
        ]


viewMasksPage : Model -> Html Msg
viewMasksPage model =
    let
        currentFrame =
            List.head (List.drop model.currentFrameIndex model.frames)

        selectedMaskIds =
            Dict.keys model.selectedMasks
    in
    div []
        [ h2 [ class "text-2xl font-bold mb-4" ] [ text "Select Masks" ]
        , case currentFrame of
            Just frame ->
                div []
                    [ div [ class "mb-4 text-sm text-gray-600" ]
                        [ text ("Frame " ++ String.fromInt (model.currentFrameIndex + 1) ++ " of " ++ String.fromInt (List.length model.frames))
                        , text (" | PK: " ++ frame.pk)
                        , text (" | " ++ String.fromInt (List.length model.masks) ++ " masks")
                        ]
                    , div [ class "flex gap-4" ]
                        [ -- Main image with overlays
                          div [ class "flex-shrink-0" ]
                            [ case model.projectId of
                                Just pid ->
                                    div [ class "relative inline-block" ]
                                        (img
                                            [ src ("/api/projects/" ++ pid ++ "/frames/" ++ frame.pk ++ "/image?scale=" ++ String.fromFloat model.maskScale)
                                            , class "block border rounded"
                                            , id "base-image"
                                            ]
                                            []
                                            :: List.indexedMap
                                                (\idx maskId ->
                                                    case List.filter (\m -> m.maskId == maskId) model.masks |> List.head of
                                                        Just mask ->
                                                            img
                                                                [ src mask.url
                                                                , class "absolute top-0 left-0 w-full h-full object-contain pointer-events-none"
                                                                , style "opacity" "0.6"
                                                                , style "filter" (maskColorFilter idx)
                                                                , style "mix-blend-mode" "screen"
                                                                ]
                                                                []

                                                        Nothing ->
                                                            text ""
                                                )
                                                selectedMaskIds
                                        )

                                Nothing ->
                                    text ""
                            ]

                        -- Mask grid
                        , div [ class "flex-1" ]
                            [ div [ class "grid grid-cols-4 gap-2 max-h-96 overflow-y-auto p-2 border rounded bg-gray-50" ]
                                (List.indexedMap (viewMaskThumbnail model) model.masks)
                            ]
                        ]
                    , div [ class "mt-4 flex gap-2" ]
                        [ button
                            [ onClick (GoToPage FramesPage)
                            , class "bg-gray-200 px-4 py-2 rounded hover:bg-gray-300"
                            ]
                            [ text "Back" ]
                        , button
                            [ onClick SaveSelection
                            , disabled (Dict.isEmpty model.selectedMasks)
                            , class "bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:bg-gray-300"
                            ]
                            [ text "Save & Continue" ]
                        ]
                    ]

            Nothing ->
                text "No frame selected"
        ]


maskColorFilter : Int -> String
maskColorFilter idx =
    let
        -- Hue rotation values for distinct colors: orange, blue, green, pink, purple, etc.
        hueRotations =
            [ 0, 180, 90, 300, 270, 45, 135, 225, 315, 30 ]

        hue =
            Maybe.withDefault 0 (List.head (List.drop (modBy 10 idx) hueRotations))
    in
    "sepia(1) saturate(5) hue-rotate(" ++ String.fromInt hue ++ "deg)"


viewMaskThumbnail : Model -> Int -> MaskMeta -> Html Msg
viewMaskThumbnail model idx mask =
    let
        isSelected =
            Dict.member mask.maskId model.selectedMasks

        selectedIndex =
            List.indexedMap Tuple.pair (Dict.keys model.selectedMasks)
                |> List.filter (\( _, id ) -> id == mask.maskId)
                |> List.head
                |> Maybe.map Tuple.first

        currentLabel =
            Dict.get mask.maskId model.selectedMasks |> Maybe.withDefault 1
    in
    div
        [ class
            ("relative cursor-pointer border-2 rounded overflow-hidden "
                ++ (if isSelected then
                        "border-blue-500 ring-2 ring-blue-300"

                    else
                        "border-gray-200 hover:border-gray-400"
                   )
            )
        , onClick (ToggleMask mask.maskId)
        ]
        [ img
            [ src mask.url
            , class "w-full aspect-square object-contain bg-gray-100"
            , style "filter"
                (case selectedIndex of
                    Just i ->
                        maskColorFilter i

                    Nothing ->
                        "none"
                )
            ]
            []
        , div [ class "absolute bottom-0 left-0 right-0 bg-black bg-opacity-60 text-white text-xs p-1 flex justify-between items-center" ]
            [ span [] [ text (String.fromInt mask.maskId) ]
            , case mask.score of
                Just s ->
                    span [] [ text (String.fromFloat (toFloat (round (s * 100)) / 100)) ]

                Nothing ->
                    text ""
            ]
        , if isSelected then
            div [ class "absolute top-1 right-1" ]
                [ input
                    [ type_ "number"
                    , Html.Attributes.min "1"
                    , value (String.fromInt currentLabel)
                    , onInput (SetMaskLabel mask.maskId)
                    , stopPropagationOn "click" (D.succeed ( SetMaskLabel mask.maskId (String.fromInt currentLabel), True ))
                    , class "w-8 h-6 text-xs text-center border rounded"
                    ]
                    []
                ]

          else
            text ""
        ]


fileDecoder : D.Decoder File
fileDecoder =
    D.at [ "target", "files" ] (D.index 0 File.decoder)
